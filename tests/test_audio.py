"""WAV 解析与格式校验测试。"""
import io
import wave

import numpy as np
import pytest

from app.audio import AudioValidationError, parse_wav
from conftest import SR, build_tail_wav, rewrap, silent_wav


def test_parse_valid_wav():
    wav, ranges = build_tail_wav([
        {"start_ms": 200, "dur_ms": 300,
         "components": [(440.0, 0.6, 0.2)], "decay_rate": 12.0},
        {"start_ms": 900, "dur_ms": 300,
         "components": [(440.0, 0.5, 1.1)], "decay_rate": 12.0},
    ])
    samples = parse_wav(wav)
    assert samples.dtype == np.float32
    assert samples.shape == (int(3.0 * SR),)
    assert np.max(np.abs(samples)) <= 1.0


def test_reject_wrong_format():
    wav, _ = build_tail_wav([
        {"start_ms": 200, "dur_ms": 300,
         "components": [(440.0, 0.6, 0.0)], "decay_rate": 12.0},
        {"start_ms": 900, "dur_ms": 300,
         "components": [(440.0, 0.5, 0.0)], "decay_rate": 12.0},
    ])
    with wave.open(io.BytesIO(wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    with pytest.raises(AudioValidationError):
        parse_wav(rewrap(pcm, nch=2, sw=2, fr=SR, nframes=len(pcm) // 4))
    with pytest.raises(AudioValidationError):
        parse_wav(rewrap(pcm, nch=1, sw=1, fr=SR, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError):
        parse_wav(rewrap(pcm, nch=1, sw=2, fr=8000, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError):
        parse_wav(b"not a wav file")


def test_reject_duration_over_8s():
    with pytest.raises(AudioValidationError):
        parse_wav(silent_wav(9.0))


def test_reject_empty_wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"")
    with pytest.raises(AudioValidationError):
        parse_wav(buf.getvalue())
