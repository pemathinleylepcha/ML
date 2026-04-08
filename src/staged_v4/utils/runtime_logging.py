from __future__ import annotations

import json
import logging
import os
import socket as socketlib
import socket
import sys
import time
import traceback
import urllib.parse
import urllib.request
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


def memory_guard_state(
    min_available_mb: float = 4096.0,
    critical_available_mb: float = 2048.0,
) -> dict[str, Any]:
    snapshot = runtime_snapshot()
    available_mb = snapshot.get("system_mem_available_mb")
    state = "unknown"
    if available_mb is not None:
        if available_mb < critical_available_mb:
            state = "critical"
        elif available_mb < min_available_mb:
            state = "low"
        else:
            state = "ok"
    return {
        "state": state,
        "available_mb": available_mb,
        "min_available_mb": float(min_available_mb),
        "critical_available_mb": float(critical_available_mb),
        "snapshot": snapshot,
    }


def guard_worker_budget(
    requested_workers: int,
    min_available_mb: float = 4096.0,
    critical_available_mb: float = 2048.0,
) -> tuple[int, dict[str, Any]]:
    guard = memory_guard_state(
        min_available_mb=min_available_mb,
        critical_available_mb=critical_available_mb,
    )
    effective_workers = int(requested_workers)
    if effective_workers <= 1:
        return max(1, effective_workers), guard
    if guard["state"] == "critical":
        effective_workers = 1
    elif guard["state"] == "low":
        effective_workers = min(effective_workers, 2)
    return max(1, effective_workers), guard


def enforce_memory_guard(
    logger: logging.Logger,
    status_file: str | None,
    stage: str,
    *,
    min_available_mb: float = 4096.0,
    critical_available_mb: float = 2048.0,
    details: dict[str, Any] | None = None,
    raise_on_critical: bool = True,
) -> dict[str, Any]:
    guard = memory_guard_state(
        min_available_mb=min_available_mb,
        critical_available_mb=critical_available_mb,
    )
    payload = {
        "state": "running",
        "stage": "memory_guard",
        "details": {
            "target_stage": stage,
            **(details or {}),
            "memory_guard_state": guard["state"],
            "memory_guard_available_mb": guard["available_mb"],
            "memory_guard_min_available_mb": guard["min_available_mb"],
            "memory_guard_critical_available_mb": guard["critical_available_mb"],
        },
    }
    if guard["state"] == "low":
        logger.warning(
            "stage=%s memory_guard=low available_mb=%s min_available_mb=%s",
            stage,
            guard["available_mb"],
            guard["min_available_mb"],
        )
        write_status(status_file, payload)
    elif guard["state"] == "critical":
        logger.error(
            "stage=%s memory_guard=critical available_mb=%s critical_available_mb=%s",
            stage,
            guard["available_mb"],
            guard["critical_available_mb"],
        )
        write_status(status_file, payload)
        if raise_on_critical:
            raise MemoryError(
                f"Memory guard tripped at stage={stage}: available_mb={guard['available_mb']} "
                f"< critical_available_mb={guard['critical_available_mb']}"
            )
    return guard


def _status_event_path(status_file: str | None) -> Path | None:
    if not status_file:
        return None
    path = Path(status_file)
    return path.with_name(f"{path.stem}_events.jsonl")


def append_status_event(status_file: str | None, payload: dict[str, Any]) -> None:
    event_path = _status_event_path(status_file)
    if event_path is None:
        return
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = {"runtime": runtime_snapshot(), **payload}
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, default=str))
        handle.write("\n")


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


def _pushgateway_settings() -> dict[str, Any] | None:
    enabled = os.environ.get("ALGOC2_PUSHGATEWAY_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    return {
        "url": os.environ.get("ALGOC2_PUSHGATEWAY_URL", "http://127.0.0.1:9091").rstrip("/"),
        "job": os.environ.get("ALGOC2_PUSHGATEWAY_JOB", "algoc2_training"),
        "instance": os.environ.get("ALGOC2_PUSHGATEWAY_INSTANCE", socket.gethostname()),
        "timeout_sec": float(os.environ.get("ALGOC2_PUSHGATEWAY_TIMEOUT_SEC", "2.0")),
    }


def _metric_label_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_metric_line(name: str, value: float, labels: dict[str, Any]) -> str:
    label_text = ",".join(f'{key}="{_metric_label_escape(raw)}"' for key, raw in labels.items() if raw is not None)
    return f"{name}{{{label_text}}} {value}"


def _status_run_name(status_file: str | None) -> str:
    if not status_file:
        return "algoc2"
    try:
        return Path(status_file).parent.name or "algoc2"
    except Exception:
        return "algoc2"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _render_pushgateway_metrics(status_file: str | None, payload: dict[str, Any]) -> bytes:
    runtime = payload.get("runtime", {}) if isinstance(payload.get("runtime"), dict) else {}
    details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
    progress = payload.get("progress", {}) if isinstance(payload.get("progress"), dict) else {}
    run_name = _status_run_name(status_file)
    stage = str(payload.get("stage", "unknown"))
    state = str(payload.get("state", "unknown"))
    hostname = str(runtime.get("hostname", socket.gethostname()))
    fold = payload.get("fold", details.get("fold"))
    fold_total = payload.get("fold_total", details.get("fold_total"))
    epoch = details.get("epoch")
    base_labels = {"run": run_name, "stage": stage, "state": state, "host": hostname}
    lines = [
        "# TYPE algoc2_training_status gauge",
        _render_metric_line("algoc2_training_status", 1.0, base_labels),
        "# TYPE algoc2_training_last_update_unixtime gauge",
        _render_metric_line("algoc2_training_last_update_unixtime", float(time.time()), {"run": run_name, "host": hostname}),
    ]

    current = _maybe_float(progress.get("current"))
    total = _maybe_float(progress.get("total"))
    ratio = _maybe_float(progress.get("ratio"))
    if current is not None:
        lines.extend(
            [
                "# TYPE algoc2_training_progress_current gauge",
                _render_metric_line("algoc2_training_progress_current", current, {"run": run_name, "stage": stage, "host": hostname}),
            ]
        )
    if total is not None:
        lines.extend(
            [
                "# TYPE algoc2_training_progress_total gauge",
                _render_metric_line("algoc2_training_progress_total", total, {"run": run_name, "stage": stage, "host": hostname}),
            ]
        )
    if ratio is not None:
        lines.extend(
            [
                "# TYPE algoc2_training_progress_ratio gauge",
                _render_metric_line("algoc2_training_progress_ratio", ratio, {"run": run_name, "stage": stage, "host": hostname}),
            ]
        )

    if fold is not None:
        fold_value = _maybe_float(fold)
        if fold_value is not None:
            lines.extend(
                [
                    "# TYPE algoc2_training_fold gauge",
                    _render_metric_line("algoc2_training_fold", fold_value, {"run": run_name, "host": hostname}),
                ]
            )
    if fold_total is not None:
        fold_total_value = _maybe_float(fold_total)
        if fold_total_value is not None:
            lines.extend(
                [
                    "# TYPE algoc2_training_fold_total gauge",
                    _render_metric_line("algoc2_training_fold_total", fold_total_value, {"run": run_name, "host": hostname}),
                ]
            )
    if epoch is not None:
        epoch_value = _maybe_float(epoch)
        if epoch_value is not None:
            lines.extend(
                [
                    "# TYPE algoc2_training_epoch gauge",
                    _render_metric_line("algoc2_training_epoch", epoch_value, {"run": run_name, "stage": stage, "host": hostname}),
                ]
            )

    runtime_metric_map = {
        "algoc2_training_runtime_rss_mb": runtime.get("rss_mb"),
        "algoc2_training_runtime_vms_mb": runtime.get("vms_mb"),
        "algoc2_training_runtime_cpu_percent": runtime.get("cpu_percent"),
        "algoc2_training_runtime_threads": runtime.get("num_threads"),
        "algoc2_training_system_cpu_percent": runtime.get("system_cpu_percent"),
        "algoc2_training_system_mem_percent": runtime.get("system_mem_percent"),
        "algoc2_training_system_mem_available_mb": runtime.get("system_mem_available_mb"),
    }
    for metric_name, raw_value in runtime_metric_map.items():
        value = _maybe_float(raw_value)
        if value is None:
            continue
        lines.extend(
            [
                f"# TYPE {metric_name} gauge",
                _render_metric_line(metric_name, value, {"run": run_name, "host": hostname}),
            ]
        )

    throughput_metric_map = {
        "algoc2_training_stage_batches": details.get("batches"),
        "algoc2_training_stage_elapsed_sec": details.get("elapsed_sec"),
        "algoc2_training_stage_batches_per_sec": details.get("batches_per_sec"),
        "algoc2_training_stage_batch_size": details.get("batch_size"),
    }
    for metric_name, raw_value in throughput_metric_map.items():
        value = _maybe_float(raw_value)
        if value is None:
            continue
        lines.extend(
            [
                f"# TYPE {metric_name} gauge",
                _render_metric_line(metric_name, value, {"run": run_name, "stage": stage, "host": hostname}),
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _publish_status_to_pushgateway(status_file: str | None, payload: dict[str, Any]) -> None:
    settings = _pushgateway_settings()
    if settings is None:
        return
    job = urllib.parse.quote(str(settings["job"]), safe="")
    instance = urllib.parse.quote(str(settings["instance"]), safe="")
    url = f"{settings['url']}/metrics/job/{job}/instance/{instance}"
    body = _render_pushgateway_metrics(status_file, payload)
    request = urllib.request.Request(url, data=body, method="PUT")
    request.add_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
    with urllib.request.urlopen(request, timeout=settings["timeout_sec"]) as response:
        response.read()


def write_status(status_file: str | None, payload: dict[str, Any]) -> None:
    merged = {"runtime": runtime_snapshot(), **payload}
    if status_file:
        path = Path(status_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    append_status_event(status_file, payload)
    try:
        _publish_status_to_redis(merged)
    except Exception:
        pass
    try:
        _publish_status_to_pushgateway(status_file, merged)
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


def log_throughput(
    logger: logging.Logger,
    stage: str,
    batches: int,
    elapsed_sec: float,
    *,
    status_file: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    safe_batches = int(max(0, batches))
    safe_elapsed = float(max(elapsed_sec, 1e-9))
    payload = {
        "stage": stage,
        "batches": safe_batches,
        "elapsed_sec": safe_elapsed,
        "batches_per_sec": float(safe_batches / safe_elapsed) if safe_batches else 0.0,
        **extra,
    }
    logger.info("state=throughput %s", json.dumps(payload, default=str))
    write_status(
        status_file,
        {
            "state": "running",
            "stage": stage,
            "progress": {
                "current": safe_batches,
                "total": safe_batches,
                "ratio": 1.0 if safe_batches else 0.0,
            },
            "details": payload,
            "fold": extra.get("fold"),
            "fold_total": extra.get("fold_total"),
        },
    )
    append_status_event(status_file, {"state": "throughput", "stage": stage, "details": payload})
    return payload
