"""
Asynchronous Provenance I/O Engine.

Implements lazy asynchronous writing of provenance data to decouple
capture from the inference critical path. Inspired by the lazy async
I/O approach in DataStates-LLM (Nicolae et al.) for checkpoint I/O.

Key design:
- Double-buffered: capture fills one buffer while previous flushes to disk
- Non-blocking: I/O happens in background thread pool
- Lazy flush: batches writes to amortize I/O overhead
- Backpressure: drops old data if queue is full (configurable)
"""

import os
import time
import threading
import queue
import pickle
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
import logging

from difftrace.core.compressor import CompressedTrajectory
from difftrace.utils.config import IOConfig, IOMode

logger = logging.getLogger(__name__)


@dataclass
class IOStats:
    """I/O performance statistics."""
    total_writes: int = 0
    total_bytes_written: int = 0
    total_write_time_ns: int = 0
    queue_drops: int = 0
    max_queue_depth: int = 0
    flush_count: int = 0

    @property
    def avg_write_latency_ms(self) -> float:
        if self.total_writes == 0:
            return 0.0
        return (self.total_write_time_ns / self.total_writes) / 1e6

    @property
    def throughput_mb_s(self) -> float:
        if self.total_write_time_ns == 0:
            return 0.0
        return (self.total_bytes_written / 1e6) / (self.total_write_time_ns / 1e9)


class AsyncProvenanceWriter:
    """
    Asynchronous I/O engine for provenance data.

    Supports two modes:
    - SYNC: Blocking writes (for debugging or when I/O latency is not critical)
    - ASYNC: Non-blocking writes with double buffering and lazy flush
    """

    def __init__(
        self,
        config: IOConfig,
        output_dir: Path,
        on_write_complete: Optional[Callable[[str, bool], None]] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._on_write_complete = on_write_complete
        self._stats = IOStats()
        self._lock = threading.Lock()

        # Async mode resources
        self._write_queue: Optional[queue.Queue] = None
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()

        # Double buffer
        self._buffer_a: List[tuple] = []  # (request_id, data)
        self._buffer_b: List[tuple] = []
        self._active_buffer = "a"
        self._buffer_lock = threading.Lock()

        if config.mode == IOMode.ASYNC:
            self._start_workers()

    def _start_workers(self):
        """Start background I/O worker threads."""
        self._write_queue = queue.Queue(maxsize=self.config.max_queue_size)
        for i in range(self.config.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"difftrace-io-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        # Lazy flush thread
        flush_thread = threading.Thread(
            target=self._flush_loop,
            name="difftrace-flush",
            daemon=True,
        )
        flush_thread.start()
        self._workers.append(flush_thread)
        logger.info(
            f"Started {self.config.num_workers} async I/O workers + flush thread"
        )

    def write_trajectory(
        self,
        compressed: CompressedTrajectory,
    ) -> bool:
        """
        Write a compressed trajectory to the provenance store.

        In ASYNC mode, this is non-blocking (enqueues for background write).
        In SYNC mode, this blocks until the write is complete.
        """
        if self.config.mode == IOMode.SYNC:
            return self._write_sync(compressed)
        else:
            return self._write_async(compressed)

    def _write_sync(self, compressed: CompressedTrajectory) -> bool:
        """Synchronous (blocking) write."""
        t_start = time.time_ns()
        try:
            data = self._serialize(compressed)
            filepath = self._get_filepath(compressed.request_id)
            with open(filepath, "wb") as f:
                f.write(data)

            t_end = time.time_ns()
            with self._lock:
                self._stats.total_writes += 1
                self._stats.total_bytes_written += len(data)
                self._stats.total_write_time_ns += (t_end - t_start)

            if self._on_write_complete:
                self._on_write_complete(compressed.request_id, True)
            return True

        except Exception as e:
            logger.error(f"Sync write failed for {compressed.request_id}: {e}")
            if self._on_write_complete:
                self._on_write_complete(compressed.request_id, False)
            return False

    def _write_async(self, compressed: CompressedTrajectory) -> bool:
        """Non-blocking async write. Enqueues data for background processing."""
        try:
            data = self._serialize(compressed)

            with self._buffer_lock:
                buf = (
                    self._buffer_a
                    if self._active_buffer == "a"
                    else self._buffer_b
                )
                buf.append((compressed.request_id, data))

            # Track max queue depth
            with self._lock:
                depth = len(self._buffer_a) + len(self._buffer_b)
                self._stats.max_queue_depth = max(
                    self._stats.max_queue_depth, depth
                )

            return True

        except Exception as e:
            logger.error(f"Async enqueue failed for {compressed.request_id}: {e}")
            return False

    def _flush_loop(self):
        """Periodically flush the active buffer to the write queue."""
        interval_s = self.config.flush_interval_ms / 1000.0
        while not self._shutdown_event.is_set():
            time.sleep(interval_s)
            self._do_flush()

    def _do_flush(self):
        """Swap buffers and enqueue the filled buffer for writing."""
        with self._buffer_lock:
            if self._active_buffer == "a":
                to_flush = self._buffer_a
                self._buffer_a = []
                self._active_buffer = "b"
            else:
                to_flush = self._buffer_b
                self._buffer_b = []
                self._active_buffer = "a"

        if not to_flush:
            return

        with self._lock:
            self._stats.flush_count += 1

        for request_id, data in to_flush:
            try:
                if self._write_queue is not None:
                    self._write_queue.put_nowait((request_id, data))
            except queue.Full:
                with self._lock:
                    self._stats.queue_drops += 1
                logger.warning(
                    f"Write queue full, dropping provenance for {request_id}"
                )

    def _worker_loop(self):
        """Background worker: dequeue and write to disk."""
        while not self._shutdown_event.is_set():
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:  # Shutdown sentinel
                break

            request_id, data = item
            t_start = time.time_ns()
            success = False
            try:
                filepath = self._get_filepath(request_id)
                with open(filepath, "wb") as f:
                    f.write(data)
                success = True
            except Exception as e:
                logger.error(f"Background write failed for {request_id}: {e}")

            t_end = time.time_ns()
            with self._lock:
                self._stats.total_writes += 1
                self._stats.total_bytes_written += len(data)
                self._stats.total_write_time_ns += (t_end - t_start)

            if self._on_write_complete:
                self._on_write_complete(request_id, success)

            self._write_queue.task_done()

    def flush_and_wait(self):
        """Force flush all pending writes and wait for completion."""
        self._do_flush()
        if self._write_queue is not None:
            self._write_queue.join()

    def shutdown(self, timeout: float = 10.0):
        """Gracefully shut down all I/O workers."""
        logger.info("Shutting down DiffTrace I/O engine...")
        self._shutdown_event.set()
        self._do_flush()

        # Send shutdown sentinels
        if self._write_queue is not None:
            for _ in range(self.config.num_workers):
                try:
                    self._write_queue.put_nowait(None)
                except queue.Full:
                    pass

        for t in self._workers:
            t.join(timeout=timeout)

        logger.info(
            f"I/O shutdown complete. Stats: {self._stats.total_writes} writes, "
            f"{self._stats.total_bytes_written / 1e6:.1f} MB, "
            f"avg latency {self._stats.avg_write_latency_ms:.2f} ms"
        )

    @property
    def stats(self) -> IOStats:
        with self._lock:
            return IOStats(
                total_writes=self._stats.total_writes,
                total_bytes_written=self._stats.total_bytes_written,
                total_write_time_ns=self._stats.total_write_time_ns,
                queue_drops=self._stats.queue_drops,
                max_queue_depth=self._stats.max_queue_depth,
                flush_count=self._stats.flush_count,
            )

    def _get_filepath(self, request_id: str) -> Path:
        """Generate filepath for a provenance record."""
        # Sanitize request_id for filesystem
        safe_id = request_id.replace("/", "_").replace("\\", "_")
        return self.output_dir / f"{safe_id}.dtrace"

    @staticmethod
    def _serialize(compressed: CompressedTrajectory) -> bytes:
        """Serialize a compressed trajectory to bytes."""
        return pickle.dumps(compressed, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(data: bytes) -> CompressedTrajectory:
        """Deserialize a compressed trajectory from bytes."""
        return pickle.loads(data)
