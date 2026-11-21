"""合成测试 WAV 与 TestClient 夹具。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import Store

SR = 16_000


def damped_tone(
    t: np.ndarray,
    freq: float,
    amp: float,
    phase: float,
    decay_rate: float,
) -> np.ndarray:
    """单分量衰减正弦：A·exp(-λt)·cos(2πft+φ)。"""
    return amp * np.exp(-decay_rate * t) * np.cos(
        2.0 * np.pi * freq * t + phase
    )


def build_tail_wav(
    segments: list[dict],
    sr: int = SR,
    noise: float = 0.0015,
    seed: int = 11,
    total_span: float = 3.0,
    sample_width: int = 2,
    channels: int = 1,
) -> tuple[bytes, list[tuple[int, int]]]:
    """合成含两段敲击尾音的 WAV。

    segments: 每段 {"start_ms", "dur_ms", "components": [(f, amp, phase)],
                    "decay_rate"}，两段共用频率与衰减率由调用方自行保证。
    返回 (wav 字节, [(start, end), ...] 采样点区间)。
    """
    rng = np.random.default_rng(seed)
    n = int(round(total_span * sr))
    x = rng.standard_normal(n).astype(np.float64) * noise
    ranges: list[tuple[int, int]] = []
    for seg in segments:
        start = int(round(seg["start_ms"] / 1000.0 * sr))
        length = int(round(seg["dur_ms"] / 1000.0 * sr))
        end = start + length
        t = np.arange(length, dtype=np.float64) / sr
        wave_seg = np.zeros(length, dtype=np.float64)
        for f, amp, phase in seg["components"]:
            wave_seg += damped_tone(t, f, amp, phase, seg["decay_rate"])
        x[start:end] += wave_seg
        ranges.append((start, end))

    x = np.clip(x, -1.0, 1.0)
    if sample_width == 2:
        pcm = (x * 32767).astype("<i2").tobytes()
        sw = 2
    else:  # 8-bit unsigned PCM
        pcm = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8).tobytes()
        sw = 1
    if channels == 2 and sample_width == 2:
        pcm = np.repeat((x * 32767).astype("<i2"), 2).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue(), ranges


def silent_wav(duration_sec: float = 2.0, sr: int = SR) -> bytes:
    pcm = np.zeros(int(duration_sec * sr), dtype="<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def rewrap(pcm: bytes, *, nch: int, sw: int, fr: int, nframes: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(fr)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.store", Store(tmp_path))
    return TestClient(app)


def post_analyze(client: TestClient, wav: bytes, ranges, band=(100.0, 2000.0)):
    (s1, e1), (s2, e2) = ranges
    return client.post(
        "/api/analyze",
        data={
            "seg1_start": s1, "seg1_end": e1,
            "seg2_start": s2, "seg2_end": e2,
            "band_lo": band[0], "band_hi": band[1],
        },
        files={"wav": ("tap.wav", wav, "audio/wav")},
    )
