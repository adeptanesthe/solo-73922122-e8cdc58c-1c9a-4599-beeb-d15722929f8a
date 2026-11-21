"""共振核对服务与结果存储。

- 输入参数（段位置、频带）不合法时抛 ResonanceError / AudioValidationError，
  由 API 层返回 400，且完全不触碰已保存的上一次结果；
- 分析完成后无论结论是 single / dual / unconfirmed 都写入存储：
  unconfirmed 同样覆盖旧结论，相当于「撤下旧结论并说明原因」。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import SAMPLE_RATE, parse_wav
from .resonance import (
    BAND_MAX_HZ,
    BAND_MIN_HZ,
    SEG_MAX_MS,
    SEG_MIN_MS,
    analyze,
)

MAX_UPLOAD_BYTES = 16_000 * 2 * 8 + 4096  # 8 秒 16k/16bit/单声道 + WAV 头余量


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_result: dict[str, Any] | None = None

    def _save_wav(self, raw: bytes) -> str:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        name = f"tap_{digest}.wav"
        path = self.uploads_dir / name
        if not path.exists():
            path.write_bytes(raw)
        return f"/data/uploads/{name}"

    def analyze(
        self,
        wav_raw: bytes,
        seg1: tuple[int, int],
        seg2: tuple[int, int],
        f_lo: float,
        f_hi: float,
    ) -> dict[str, Any]:
        # 先完整解析与分析，任何一步抛错都不会写入状态
        samples = parse_wav(wav_raw)
        result = analyze(samples, [seg1, seg2], float(f_lo), float(f_hi))

        url = self._save_wav(wav_raw)
        result["id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["audio_url"] = url
        result["sample_rate"] = SAMPLE_RATE
        result["segment_rule_ms"] = [SEG_MIN_MS, SEG_MAX_MS]
        result["band_rule_hz"] = [BAND_MIN_HZ, BAND_MAX_HZ]

        with self._lock:
            self._last_result = result
        return result

    def get_last(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_result is None else dict(self._last_result)

    def clear(self) -> None:
        with self._lock:
            self._last_result = None
