"""生成演示用 WAV（16kHz/16bit/单声道），供手工冒烟测试。

用法：python -I scripts/gen_demo.py <输出目录>
生成同一模具两次敲击尾音的三类录音：
- tail_single.wav  单频共振（应报单频）
- tail_dual.wav    双频共振、相隔 283Hz（应报双频）
- tail_noise.wav   无规则噪声尾音（应无法确认）
同时打印每段的采样点区间与建议频带，方便在核对页直接填写。
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16_000
SEG1_START_MS = 200
SEG2_START_MS = 1100
DUR_MS = 320
LAM = 14.0
F1, F2 = 432.0, 715.0


def _seg(t, components):
    out = np.zeros_like(t)
    for f, amp, phase in components:
        out += amp * np.exp(-LAM * t) * np.cos(2 * np.pi * f * t + phase)
    return out


def build(components_pair, noise_amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(3.0 * SR)
    x = rng.standard_normal(n).astype(np.float64) * noise_amp
    length = int(round(DUR_MS / 1000.0 * SR))
    t = np.arange(length, dtype=np.float64) / SR
    for start_ms, comps in [(SEG1_START_MS, components_pair[0]),
                            (SEG2_START_MS, components_pair[1])]:
        start = int(round(start_ms / 1000.0 * SR))
        x[start:start + length] += _seg(t, comps)
    return np.clip(x, -1.0, 1.0)


def save(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save(out / "tail_single.wav", build(
        ([(F1, 0.65, 0.3)], [(F1, 0.50, 1.7)]), noise_amp=0.0012, seed=1))
    save(out / "tail_dual.wav", build(
        ([(F1, 0.55, 0.2), (F2, 0.35, 1.0)],
         [(F1, 0.45, 1.8), (F2, 0.40, 2.4)]), noise_amp=0.0012, seed=2))

    rng = np.random.default_rng(3)
    save(out / "tail_noise.wav",
         np.clip(rng.standard_normal(int(3.0 * SR)) * 0.05, -1, 1))

    s1 = int(round(SEG1_START_MS / 1000.0 * SR))
    s2 = int(round(SEG2_START_MS / 1000.0 * SR))
    length = int(round(DUR_MS / 1000.0 * SR))
    print("written to", out)
    print(f"第 1 段采样点: {s1} ~ {s1 + length}（{DUR_MS} ms）")
    print(f"第 2 段采样点: {s2} ~ {s2 + length}（{DUR_MS} ms）")
    print("待查频带建议: 100 ~ 2000 Hz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
