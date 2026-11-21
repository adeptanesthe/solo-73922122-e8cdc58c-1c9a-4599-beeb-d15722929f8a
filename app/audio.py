"""WAV 解析与基础校验。

输入要求（沿用现有格式）：16 kHz、16-bit PCM、单声道 WAV，时长不超过 8 秒。
本模块只负责把 WAV 解析成 float32、范围 [-1, 1] 的单声道波形，
尾音频段的圈选与共振拟合分别在 service / resonance 模块完成。
"""
from __future__ import annotations

import io
import wave

import numpy as np

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1
MAX_DURATION_SEC = 8.0


class AudioValidationError(ValueError):
    """音频文件不符合约定格式或长度。"""


def parse_wav(data: bytes) -> np.ndarray:
    """校验并解析 WAV，返回 float32、范围 [-1, 1] 的单声道波形。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioValidationError(f"无法解析 WAV 文件：{exc}") from exc

    if n_channels != CHANNELS:
        raise AudioValidationError(f"只支持单声道 WAV，当前声道数：{n_channels}")
    if sample_width != SAMPLE_WIDTH:
        raise AudioValidationError(
            f"只支持 16-bit PCM，当前采样位深：{sample_width * 8}-bit"
        )
    if frame_rate != SAMPLE_RATE:
        raise AudioValidationError(
            f"只支持 {SAMPLE_RATE} Hz 采样率，当前：{frame_rate} Hz"
        )

    duration = n_frames / frame_rate
    if duration > MAX_DURATION_SEC + 1e-9:
        raise AudioValidationError(
            f"录音时长 {duration:.2f}s 超过上限 {MAX_DURATION_SEC:.0f}s"
        )
    if n_frames == 0:
        raise AudioValidationError("录音为空，没有任何采样点")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.shape[0] != n_frames:
        raise AudioValidationError("WAV 数据块长度与声明帧数不一致，文件可能损坏")
    return samples
