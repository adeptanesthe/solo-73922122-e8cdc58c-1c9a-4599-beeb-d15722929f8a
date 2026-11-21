"""衰减正弦拟合与单/双频判定的算法测试。"""
import numpy as np
import pytest

from app.audio import SAMPLE_RATE, parse_wav
from app.resonance import (
    ABLATION_DELTA,
    HOLDOUT_LIMIT,
    ResonanceError,
    analyze,
    make_segments,
    validate_band,
)
from conftest import build_tail_wav

# 合成参数：两段共用频率与衰减率，振幅/相位各自不同
F1 = 432.0
F2 = 715.0
LAM = 14.0  # τ ≈ 71.4 ms
DUR_MS = 320
SEG1_START = 200
SEG2_START = 1100


def _single_wav(noise=0.0012):
    return build_tail_wav([
        {"start_ms": SEG1_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.65, 0.3)], "decay_rate": LAM},
        {"start_ms": SEG2_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.50, 1.7)], "decay_rate": LAM},
    ], noise=noise)


def _dual_wav(noise=0.0012):
    return build_tail_wav([
        {"start_ms": SEG1_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.55, 0.2), (F2, 0.35, 1.0)],
         "decay_rate": LAM},
        {"start_ms": SEG2_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.45, 1.8), (F2, 0.40, 2.4)],
         "decay_rate": LAM},
    ], noise=noise)


def _analyze(wav, ranges, band=(100.0, 2000.0)):
    return analyze(parse_wav(wav), ranges, band[0], band[1])


def test_single_tone_passes_and_recovers_parameters():
    wav, ranges = _single_wav()
    r = _analyze(wav, ranges)
    assert r["status"] == "single"
    assert len(r["frequencies_hz"]) == 1
    assert r["frequencies_hz"][0] == pytest.approx(F1, abs=1.5)
    assert r["decay_rate"] == pytest.approx(LAM, rel=0.15)
    # 两段 holdout 误差都 ≤10%
    assert all(x <= HOLDOUT_LIMIT + 1e-12 for x in r["holdout_ratios"])
    # 单频通过时不应再启用双频
    assert r["dual"] is None
    # 每段各自的振幅/相位都被拟合
    comps = r["single"]["components"]
    assert {c["segment"] for c in comps} == {0, 1}
    amps = sorted(c["amplitude"] for c in comps)
    assert amps[0] == pytest.approx(0.50, abs=0.05)
    assert amps[1] == pytest.approx(0.65, abs=0.05)


def test_single_tone_band_restricts_frequency():
    # 频带把 432Hz 排除在外：只能在 500~900 内拟合，单频必然不通过
    wav, ranges = _single_wav()
    r = _analyze(wav, ranges, band=(500.0, 900.0))
    assert r["status"] == "unconfirmed"
    assert r["single"]["frequencies_hz"][0] >= 500.0
    assert any("误差能量占比超过 10%" in x for x in r["reasons"])


def test_dual_tone_passes_full_criteria():
    wav, ranges = _dual_wav()
    r = _analyze(wav, ranges)
    assert r["status"] == "dual", r["reasons"]
    freqs = r["frequencies_hz"]
    assert freqs[0] == pytest.approx(F1, abs=2.0)
    assert freqs[1] == pytest.approx(F2, abs=2.0)
    assert freqs[1] - freqs[0] >= 5.0
    assert r["decay_rate"] == pytest.approx(LAM, rel=0.2)
    assert all(x <= HOLDOUT_LIMIT + 1e-12 for x in r["holdout_ratios"])
    # 消融：去掉任一分量，两段误差占比都增加 ≥10 个百分点
    for ab in r["dual"]["ablation"]:
        assert all(d >= ABLATION_DELTA - 1e-12 for d in ab["delta"])


def test_dual_tone_weak_component_is_rejected():
    # 第二分量只在第 1 段出现、第 2 段几乎不存在：去掉它时第 2 段误差
    # 增量约为 0，不满足「两段都增加 ≥10 个百分点」，双频必须被否决
    wav, ranges = build_tail_wav([
        {"start_ms": SEG1_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.60, 0.2), (F2, 0.30, 1.0)],
         "decay_rate": LAM},
        {"start_ms": SEG2_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.55, 1.8)],
         "decay_rate": LAM},
    ])
    r = _analyze(wav, ranges)
    assert r["status"] == "unconfirmed"
    assert r["dual"] is not None  # 双频确实被尝试
    # 核心判据：存在某分量对某段的消融增量不足 10pp，双频因此被否决
    failed = any(
        any(d < ABLATION_DELTA - 1e-12 for d in ab["delta"])
        for ab in r["dual"]["ablation"]
    )
    assert failed
    assert any("独立贡献" in x for x in r["reasons"])


def test_dual_tone_minimum_5hz_gap_supported():
    # 两个共振只相隔 5 Hz（规则允许的最小间隔），仍应能确认双频
    f_near = F1 + 5.0
    wav, ranges = build_tail_wav([
        {"start_ms": SEG1_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.5, 0.2), (f_near, 0.45, 1.0)],
         "decay_rate": LAM},
        {"start_ms": SEG2_START, "dur_ms": DUR_MS,
         "components": [(F1, 0.45, 1.8), (f_near, 0.45, 2.4)],
         "decay_rate": LAM},
    ])
    r = _analyze(wav, ranges)
    assert r["status"] == "dual", r["reasons"]
    gap = r["frequencies_hz"][1] - r["frequencies_hz"][0]
    assert gap >= 5.0 - 1e-9
    assert r["frequencies_hz"][0] == pytest.approx(F1, abs=2.5)
    assert all(x <= HOLDOUT_LIMIT + 1e-12 for x in r["holdout_ratios"])


def test_noise_only_is_unconfirmed_not_false_positive():
    rng = np.random.default_rng(3)
    n = int(3.0 * SAMPLE_RATE)
    samples = (rng.standard_normal(n) * 0.05).astype(np.float32)
    r1 = (int(0.2 * SAMPLE_RATE), int(0.52 * SAMPLE_RATE))
    r2 = (int(1.1 * SAMPLE_RATE), int(1.42 * SAMPLE_RATE))
    r = analyze(samples, [r1, r2], 100.0, 2000.0)
    assert r["status"] == "unconfirmed"
    assert r["frequencies_hz"] is None
    assert r["reasons"]  # 必须给出原因


def test_segment_duration_bounds_enforced():
    samples = np.zeros(int(3 * SAMPLE_RATE), dtype=np.float32)
    # 150 ms：低于 200ms 下限
    short = (int(0.2 * SAMPLE_RATE), int(0.35 * SAMPLE_RATE))
    ok = (int(1.0 * SAMPLE_RATE), int(1.32 * SAMPLE_RATE))
    with pytest.raises(ResonanceError, match="200"):
        make_segments(samples, [short, ok])
    # 600 ms：超过 500ms 上限
    long_seg = (int(1.0 * SAMPLE_RATE), int(1.6 * SAMPLE_RATE))
    with pytest.raises(ResonanceError, match="500"):
        make_segments(samples, [ok, long_seg])


def test_overlapping_segments_rejected():
    samples = np.zeros(int(3 * SAMPLE_RATE), dtype=np.float32)
    r1 = (int(0.2 * SAMPLE_RATE), int(0.5 * SAMPLE_RATE))
    r2 = (int(0.4 * SAMPLE_RATE), int(0.72 * SAMPLE_RATE))  # 与 r1 重叠
    with pytest.raises(ResonanceError, match="互不重叠"):
        make_segments(samples, [r1, r2])


def test_adjacent_segments_allowed():
    # 端点相接不算重叠
    wav, ranges = _single_wav()
    samples = parse_wav(wav)
    r1 = (int(0.2 * SAMPLE_RATE), int(0.52 * SAMPLE_RATE))
    r2 = (int(0.52 * SAMPLE_RATE), int(0.84 * SAMPLE_RATE))
    segs = make_segments(samples, [r1, r2])
    assert len(segs) == 2


def test_band_validation():
    assert validate_band(100.0, 2000.0) == (100.0, 2000.0)
    with pytest.raises(ResonanceError, match="100~2000"):
        validate_band(50.0, 1000.0)
    with pytest.raises(ResonanceError, match="100~2000"):
        validate_band(100.0, 2500.0)
    with pytest.raises(ResonanceError, match="下限"):
        validate_band(800.0, 800.0)


def test_silent_segment_rejected():
    samples = np.zeros(int(3 * SAMPLE_RATE), dtype=np.float32)
    r1 = (int(0.2 * SAMPLE_RATE), int(0.52 * SAMPLE_RATE))
    r2 = (int(1.1 * SAMPLE_RATE), int(1.42 * SAMPLE_RATE))
    with pytest.raises(ResonanceError, match="静音"):
        make_segments(samples, [r1, r2])


def test_fit_uses_only_first_two_thirds():
    # 段结构：fit 区恰好 2/3，holdout 区 1/3
    wav, ranges = _single_wav()
    segs = make_segments(parse_wav(wav), ranges)
    total = len(segs[0].samples)
    assert len(segs[0].fit) + len(segs[0].holdout) == total
    assert len(segs[0].fit) == pytest.approx(total * 2 / 3, abs=1)
    assert len(segs[0].holdout) >= 2


def test_frequencies_shared_but_amplitude_phase_per_segment():
    wav, ranges = _dual_wav()
    r = _analyze(wav, ranges)
    assert r["status"] == "dual"
    # 两段共用：结果里只有一对频率、一个衰减率
    assert len(r["frequencies_hz"]) == 2
    # 每段每分量各自有独立振幅/相位
    comps = r["dual"]["components"]
    assert len(comps) == 4  # 2 段 × 2 分量
    for seg_idx in (0, 1):
        seg_comps = [c for c in comps if c["segment"] == seg_idx]
        assert len(seg_comps) == 2
        assert all(c["amplitude"] > 0 for c in seg_comps)


def test_unconfirmed_reports_reasons_and_model_details():
    rng = np.random.default_rng(5)
    n = int(3.0 * SAMPLE_RATE)
    samples = (rng.standard_normal(n) * 0.04).astype(np.float32)
    r1 = (int(0.2 * SAMPLE_RATE), int(0.52 * SAMPLE_RATE))
    r2 = (int(1.1 * SAMPLE_RATE), int(1.42 * SAMPLE_RATE))
    r = analyze(samples, [r1, r2], 100.0, 2000.0)
    assert r["status"] == "unconfirmed"
    assert r["single"] is not None and r["dual"] is not None
    assert r["reasons"], "不可确认时必须说明原因"
    # 轨迹供前端画图
    assert len(r["single"]["traces"]) == 2
    assert "actual" in r["single"]["traces"][0]
    assert "prediction" in r["single"]["traces"][0]
