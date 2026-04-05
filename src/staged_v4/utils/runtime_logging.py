from __future__ import annotations

import json
import logging
import os
import socket as socketlib
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def configure_logging(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if psutil is not None:
        process = psutil.Process()
        with process.oneshot():
            snapshot.update(
                {
                    "rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                    "vms_mb": round(process.memory_info().vms / (1024 * 1024), 2),
                    "cpu_percent": process.cpu_percent(interval=None),
                    "num_threads": process.num_threads(),
                }
            )
        try:
            vm = psutil.virtual_memory()
            snapshot.update(
                {
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "system_mem_percent": vm.percent,
                    "system_mem_available_mb": round(vm.available / (1024 * 1024), 2),
                }
            )
        except Exception:
            pass
    return snapshot


def _redis_settings() -> dict[str, Any] | None:
    enabled = os.environ.get("ALGOC2_REDIS_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    return {
        "host": os.environ.get("ALGOC2_REDIS_HOST", "127.0.0.1"),
        "port": int(os.environ.get("ALGOC2_REDIS_PORT", "6379")),
        "db": int(os.environ.get("ALGOC2_REDIS_DB", "0")),
        "password": os.environ.get("ALGOC2_REDIS_PASSWORD", ""),
        "prefix": os.environ.get("ALGOC2_REDIS_PREFIX", "algoc2"),
        "history_limit": int(os.environ.get("ALGOC2_REDIS_HISTORY_LIMIT", "500")),
        "timeout_sec": float(os.environ.get("ALGOC2_REDIS_TIMEOUT_SEC", "1.5")),
    }


def _redis_pack(*parts: Any) -> bytes:
    encoded_parts = []
    for part in parts:
        if isinstance(part, bytes):
            data = part
        else:
            data = str(part).encode("utf-8")
        encoded_parts.append(b"$" + str(len(data)).encode("ascii") + b"\r\n" + data + b"\r\n")
    return b"*" + str(len(encoded_parts)).encode("ascii") + b"\r\n" + b"".join(encoded_parts)


def _redis_read_response(sock: socketlib.socket) -> bytes:
    first = sock.recv(1)
    if not first:
        raise ConnectionError("Empty response from Redis")
    if first in {b"+", b"-", b":"}:
        line = b""
        while not line.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                break
            line += chunk
        if first == b"-":
            raise ConnectionError(f"Redis error: {line[:-2].decode('utf-8', errors='replace')}")
        return line[:-2]
    if first == b"$":
        length_bytes = b""
        while not length_bytes.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                break
            length_bytes += chunk
        length = int(length_bytes[:-2])
        if length < 0:
            return b""
        remaining = length + 2
        payload = b""
        while len(payload) < remaining:
            payload += sock.recv(remaining - len(payload))
        return payload[:-2]
    if first == b"*":
        length_bytes = b""
        while not length_bytes.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                break
            length_bytes += chunk
        count = int(length_bytes[:-2])
        data = []
        for _ in range(count):
            data.append(_redis_read_response(sock))
        return b"\n".join(data)
    raise ConnectionError(f"Unsupported Redis response prefix: {first!r}")


def _redis_exec(*parts: Any) -> bytes:
    settings = _redis_settings()
    if settings is None:
        return b""
    with socketlib.create_connection((settings["host"], settings["port"]), timeout=settings["timeout_sec"]) as sock:
        if settings["password"]:
            sock.sendall(_redis_pack("AUTH", settings["password"]))
            _redis_read_response(sock)
        if settings["db"]:
            sock.sendall(_redis_pack("SELECT", settings["db"]))
            _redis_read_response(sock)
        sock.sendall(_redis_pack(*parts))
        return _redis_read_response(sock)


def _publish_status_to_redis(payload: dict[str, Any]) -> None:
    settings = _redis_settings()
    if settings is None:
        return
    encoded = json.dumps(payload, separators=(",", ":"), default=str)
    prefix = settings["prefix"]
    stage = str(payload.get("stage", "unknown"))
    latest_key = f"{prefix}:latest"
    stage_key = f"{prefix}:stage:{stage}"
    history_key = f"{prefix}:history"
    _redis_exec("SET", latest_key, encoded)
    _redis_exec("SET", stage_key, encoded)
    _redis_exec("LPUSH", history_key, encoded)
    _redis_exec("LTRIM", history_key, 0, settings["history_limit"] - 1)


def write_status(status_file: str | None, payload: dict[str, Any]) -> None:
    merged = {"runtime": runtime_snapshot(), **payload}
    if status_file:
        path = Path(status_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    try:
        _publish_status_to_redis(merged)
    except Exception:
        pass


def log_exception(logger: logging.Logger, status_file: str | None, stage: str, exc: BaseException) -> None:
    logger.exception("stage=%s failed: %s", stage, exc)
    write_status(
        status_file,
        {
            "state": "failed",
            "stage": stage,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        },
    )


@contextmanager
def stage_context(logger: logging.Logger, status_file: str | None, stage: str, **extra: Any):
    start = time.perf_counter()
    logger.info("stage=%s state=start %s", stage, json.dumps(extra, default=str))
    write_status(status_file, {"state": "running", "stage": stage, "details": extra, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    try:
        yield
    except Exception as exc:
        log_exception(logger, status_file, stage, exc)
        raise
    duration = time.perf_counter() - start
    logger.info("stage=%s state=done duration_sec=%.2f", stage, duration)
    write_status(status_file, {"state": "running", "stage": stage, "details": extra, "duration_sec": duration, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def log_progress(
    logger: logging.Logger,
    status_file: str | None,
    stage: str,
    current: int,
    total: int,
    **extra: Any,
) -> None:
    payload = {
        "state": "running",
        "stage": stage,
        "progress": {
            "current": int(current),
            "total": int(total),
            "ratio": float(current / total) if total else 0.0,
        },
        "details": extra,
    }
    logger.info("stage=%s progress=%d/%d %s", stage, current, total, json.dumps(extra, default=str))
    write_status(status_file, payload)
