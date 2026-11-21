"""模具敲击尾音的共振频率拟合与判定。

模型（衰减正弦叠加）：
    单频 x(t) = exp(-λ t) · (a·cos(2π f t) + b·sin(2π f t))
    双频 x(t) = Σ_k exp(-λ t) · (a_k·cos(2π f_k t) + b_k·sin(2π f_k t))

约定：
1. 每段只取前 2/3 样本参与拟合（fit 区），其余 1/3 作为预测核对区（holdout）；
2. 两段共用频率 {f_k} 与衰减率 λ，每段各自拟合振幅 a、b（即振幅与相位）；
3. 频率只能落在用户给定的待查频带 [f_lo, f_hi]（100~2000 Hz）内；
4. 单频模型两段 holdout 误差能量占比都 ≤10% 才通过；否则尝试双频，
   双频须满足两频相隔 ≥5 Hz、holdout 都 ≤10%，且单独去掉任一分量后
   两段误差占比都各增加至少 10 个百分点，才报双频；否则结论为不可确认。

实现上只用 numpy：频率/衰减率用自写 Nelder-Mead 做非线性优化，
固定频率与衰减率后，振幅/相位是线性最小二乘问题，直接正规方程求解。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE

# ---- 业务常量（与需求一致）-----------------------------------------------
SEG_MIN_MS = 200
SEG_MAX_MS = 500
BAND_MIN_HZ = 100.0
BAND_MAX_HZ = 2000.0
FIT_FRACTION = 2.0 / 3.0          # 每段前 2/3 参与拟合
HOLDOUT_LIMIT = 0.10              # holdout 误差能量占比上限 10%
FREQ_GAP_HZ = 5.0                 # 双频最小间隔
ABLATION_DELTA = 0.10             # 去掉分量后误差占比至少增加 10 个百分点
SILENCE_ENERGY = 1e-9             # 段平均能量低于此值视为静音，无法分析
NM_MAX_ITER = 160
NM_XTOL = 1e-6
NM_FTOL = 1e-8
# 衰减时间常数 τ=1/λ 限制在 8ms~8s，防止优化器走向无衰减/瞬时消失的退化解
TAU_MIN_SEC = 0.008
TAU_MAX_SEC = 8.0
LAMBDA_MIN = 1.0 / TAU_MAX_SEC
LAMBDA_MAX = 1.0 / TAU_MIN_SEC


class ResonanceError(ValueError):
    """段/频带参数不合法，或数据无法用于拟合。"""


@dataclass
class Segment:
    """一段圈选尾音的拟合上下文。"""

    index: int
    start_sample: int       # 该段在整段录音中的起始采样点
    samples: np.ndarray
    fit: np.ndarray          # 前 2/3 样本（参与拟合）
    holdout: np.ndarray      # 后 1/3 样本（只做预测核对）
    t_fit: np.ndarray
    t_holdout: np.ndarray
    fit_energy: float
    holdout_energy: float

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / SAMPLE_RATE * 1000.0


@dataclass
class LinearFit:
    """固定频率与衰减率后，一段尾音的线性最小二乘结果。"""

    coeffs: np.ndarray       # [a1, b1(, a2, b2)]，对应 cos/sin 振幅
    residual_energy: float   # fit 区残差能量（两段求和用于非线性优化）
    pred_fit: np.ndarray
    pred_holdout: np.ndarray
    holdout_error_ratio: float


@dataclass
class ModelResult:
    """单频或双频模型的完整结果。"""

    kind: str                # "single" / "dual"
    frequencies: list[float]
    decay_rate: float        # λ（1/s）
    tau_ms: float            # 衰减时间常数 τ=1/λ（毫秒）
    converged: bool
    fit_residual_ratio: list[float] = field(default_factory=list)
    holdout_ratio: list[float] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    ablation: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ---- 段提取与校验 ----------------------------------------------------------
def make_segments(
    samples: np.ndarray, ranges: list[tuple[int, int]]
) -> list[Segment]:
    """按 (start, end) 采样点区间切出两段尾音并做全部业务校验。"""
    if len(ranges) != 2:
        raise ResonanceError("必须圈出同一模具两次敲击的两段尾音")
    n_total = len(samples)
    clipped: list[tuple[int, int]] = []
    for idx, (start, end) in enumerate(ranges):
        if not (isinstance(start, (int, np.integer)) and
                isinstance(end, (int, np.integer))):
            raise ResonanceError(f"第 {idx + 1} 段的起止位置必须是采样点整数")
        if end <= start:
            raise ResonanceError(f"第 {idx + 1} 段结束位置必须晚于开始位置")
        if start < 0 or end > n_total:
            raise ResonanceError(f"第 {idx + 1} 段超出录音范围")
        length_ms = (end - start) / SAMPLE_RATE * 1000.0
        if not (SEG_MIN_MS - 1e-9 <= length_ms <= SEG_MAX_MS + 1e-9):
            raise ResonanceError(
                f"第 {idx + 1} 段时长 {length_ms:.0f} ms，"
                f"每段必须在 {SEG_MIN_MS}~{SEG_MAX_MS} 毫秒之间"
            )
        clipped.append((int(start), int(end)))

    (s1, e1), (s2, e2) = clipped
    if max(s1, s2) < min(e1, e2):  # 允许端点相接，不允许重叠
        raise ResonanceError("两段尾音互不重叠，请重新圈选")
    # 按时间先后重排，报告里统一叫第 1/2 段
    ordered = sorted(clipped, key=lambda r: r[0])

    segments: list[Segment] = []
    for idx, (start, end) in enumerate(ordered):
        seg_samples = samples[start:end].astype(np.float64, copy=True)
        seg_samples -= float(np.mean(seg_samples))  # 去直流，避免偏置污染正弦拟合
        n_fit = int(round(len(seg_samples) * FIT_FRACTION))
        n_fit = max(2, min(n_fit, len(seg_samples) - 1))
        fit = seg_samples[:n_fit]
        holdout = seg_samples[n_fit:]
        if len(holdout) < 2:
            raise ResonanceError(
                f"第 {idx + 1} 段过短，前 2/3 拟合后剩余样本不足以核对预测"
            )
        fit_energy = float(np.dot(fit, fit))
        holdout_energy = float(np.dot(holdout, holdout))
        if fit_energy <= SILENCE_ENERGY * len(fit) or \
                holdout_energy <= SILENCE_ENERGY * len(holdout):
            raise ResonanceError(
                f"第 {idx + 1} 段能量接近静音，无法拟合衰减正弦，"
                "请重新圈选真实尾音"
            )
        segments.append(
            Segment(
                index=idx,
                start_sample=start,
                samples=seg_samples,
                fit=fit,
                holdout=holdout,
                t_fit=np.arange(n_fit, dtype=np.float64) / SAMPLE_RATE,
                t_holdout=np.arange(
                    n_fit, len(seg_samples), dtype=np.float64
                ) / SAMPLE_RATE,
                fit_energy=fit_energy,
                holdout_energy=holdout_energy,
            )
        )
    return segments


def validate_band(f_lo: float, f_hi: float) -> tuple[float, float]:
    """校验待查频带（100~2000 Hz）。"""
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        raise ResonanceError("待查频带必须是数字")
    if f_lo < BAND_MIN_HZ - 1e-9 or f_hi > BAND_MAX_HZ + 1e-9:
        raise ResonanceError(
            f"待查频带必须落在 {BAND_MIN_HZ:.0f}~{BAND_MAX_HZ:.0f} Hz 内"
        )
    if f_lo >= f_hi:
        raise ResonanceError("频带下限必须小于上限")
    return float(f_lo), float(f_hi)


# ---- 频率候选（加窗 FFT + 抛物线插值，限定在待查频带内）-------------------
def fft_peak_candidates(
    segments: list[Segment], f_lo: float, f_hi: float, top_k: int
) -> list[float]:
    """从各段 fit 区的加窗 FFT 中提取频带内的峰频候选（亚 bin 精度）。"""
    candidates: list[float] = []
    n_target = max(len(s.fit) for s in segments)
    nfft = 1
    while nfft < n_target * 8:
        nfft *= 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SAMPLE_RATE)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    band_idx = np.where(band)[0]
    if len(band_idx) < 3:
        raise ResonanceError("待查频带过窄，FFT 无法分辨其中频率")

    for seg in segments:
        x = seg.fit - np.mean(seg.fit)
        window = np.hanning(len(x))
        spectrum = np.abs(np.fft.rfft(x * window, n=nfft))
        spec_band = spectrum[band_idx]
        # 局部极大值（频带边界处不取峰）
        peak_pos = np.where(
            (spec_band[1:-1] > spec_band[:-2]) &
            (spec_band[1:-1] >= spec_band[2:])
        )[0] + 1
        if len(peak_pos) == 0:
            peak_pos = np.array([int(np.argmax(spec_band))])
        order = peak_pos[np.argsort(spec_band[peak_pos])[::-1]][:top_k]
        for p in order:
            gi = band_idx[p]
            # 抛物线插值把峰频修到亚 bin
            y0, y1, y2 = spectrum[gi - 1], spectrum[gi], spectrum[gi + 1]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
            f = freqs[gi] + delta * (freqs[1] - freqs[0])
            f = float(np.clip(f, f_lo, f_hi))
            if all(abs(f - g) > 1.0 for g in candidates):
                candidates.append(f)
    if not candidates:
        raise ResonanceError("待查频带内未检测到任何频谱峰，无法拟合")
    return candidates


# ---- 线性最小二乘：固定 f、λ 求每段振幅/相位 ------------------------------
def _design_matrix(t: np.ndarray, freqs: np.ndarray, lam: float) -> np.ndarray:
    env = np.exp(-lam * t)
    cols = []
    for f in freqs:
        ang = 2.0 * np.pi * f * t
        cols.append(env * np.cos(ang))
        cols.append(env * np.sin(ang))
    return np.column_stack(cols)


def solve_linear(
    segments: list[Segment], freqs: np.ndarray, lam: float
) -> list[LinearFit]:
    """对每段分别求解振幅/相位（a、b），并给出 fit/holdout 预测与误差。

    设计矩阵只有 2 或 4 列，用正规方程 + 微小岭项代替 SVD-lstsq，
    在优化器上万次调用时快一个数量级以上，数值稳定性对本场景足够。
    """
    results: list[LinearFit] = []
    for seg in segments:
        A = _design_matrix(seg.t_fit, freqs, lam)
        ata = A.T @ A
        ridge = 1e-10 * float(np.trace(ata)) / ata.shape[0]
        ata = ata + ridge * np.eye(ata.shape[0])
        coeffs = np.linalg.solve(ata, A.T @ seg.fit)
        pred_fit = A @ coeffs
        residual = seg.fit - pred_fit
        A_hold = _design_matrix(seg.t_holdout, freqs, lam)
        pred_holdout = A_hold @ coeffs
        err_energy = float(np.dot(
            seg.holdout - pred_holdout, seg.holdout - pred_holdout
        ))
        results.append(
            LinearFit(
                coeffs=coeffs,
                residual_energy=float(np.dot(residual, residual)),
                pred_fit=pred_fit,
                pred_holdout=pred_holdout,
                holdout_error_ratio=err_energy / seg.holdout_energy,
            )
        )
    return results


def _total_residual(
    segments: list[Segment], freqs: np.ndarray, lam: float
) -> float:
    return float(sum(lf.residual_energy for lf in
                     solve_linear(segments, freqs, lam)))


# ---- Nelder-Mead（单纯形法，无第三方依赖）---------------------------------
def _nelder_mead(
    func, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    step_frac: float = 0.02,
) -> tuple[np.ndarray, float, bool]:
    """带边界软约束的 Nelder-Mead。返回 (最优点, 最优值, 是否收敛)。

    step_frac 控制初始单纯形相对参数自身量级的比例；频率维与衰减率维
    另有绝对步长下限，避免小尺度参数（如双频的频率间隔）的单纯形退化。
    """
    n = len(x0)
    x0 = np.clip(x0.copy(), lo, hi)
    # 最后一维约定为衰减率 λ，其余为频率（中心/间隔）维
    min_steps = np.full(n, 1.0)   # 频率维最小 1 Hz
    min_steps[-1] = 2.0           # λ 维最小 2 (1/s)
    spans = np.maximum(np.abs(x0) * 0.1, min_steps)
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += step_frac * spans[i]
        simplex.append(np.clip(x, lo, hi))
    simplex = np.array(simplex, dtype=np.float64)
    values = np.array([func(x) for x in simplex])
    converged = False

    for _ in range(NM_MAX_ITER):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        xr = np.clip(centroid + 1.0 * (centroid - worst), lo, hi)
        fr = func(xr)
        if values[0] <= fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
        elif fr < values[0]:
            xe = np.clip(centroid + 2.0 * (xr - centroid), lo, hi)
            fe = func(xe)
            if fe < fr:
                simplex[-1], values[-1] = xe, fe
            else:
                simplex[-1], values[-1] = xr, fr
        else:
            xc = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
            fc = func(xc)
            if fc < values[-1]:
                simplex[-1], values[-1] = xc, fc
            else:
                best = simplex[0]
                for k in range(1, len(simplex)):
                    simplex[k] = np.clip(
                        best + 0.5 * (simplex[k] - best), lo, hi
                    )
                values = np.array([func(x) for x in simplex])

        spread = np.max(np.abs(simplex - simplex[0]))
        fspread = np.max(np.abs(values - values[0]))
        if spread < NM_XTOL and fspread < NM_FTOL:
            converged = True
            break

    order = np.argsort(values)
    return simplex[order[0]], values[order[0]], converged


def _penalty(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """越界软惩罚（量级远大于正常残差，正常信号能量约为 O(段长)）。"""
    below = np.maximum(0.0, lo - x)
    above = np.maximum(0.0, x - hi)
    scale = np.maximum(1.0, hi - lo)
    return 1e6 * float(np.sum((below / scale) ** 2 + (above / scale) ** 2))


# ---- 单频拟合 --------------------------------------------------------------
def fit_single(
    segments: list[Segment], f_lo: float, f_hi: float
) -> ModelResult:
    lo = np.array([f_lo, LAMBDA_MIN])
    hi = np.array([f_hi, LAMBDA_MAX])
    f_candidates = fft_peak_candidates(segments, f_lo, f_hi, top_k=3)

    def cost(x: np.ndarray) -> float:
        f, lam = x
        if not (f_lo <= f <= f_hi and LAMBDA_MIN <= lam <= LAMBDA_MAX):
            return _penalty(x, lo, hi)
        return _total_residual(segments, np.array([f]), lam) + _penalty(x, lo, hi)

    best = None  # (value, x, converged)
    for f0 in f_candidates[:3]:
        for lam0 in (10.0, 80.0):
            x, val, ok = _nelder_mead(
                cost, np.array([f0, lam0]), lo, hi, step_frac=0.02
            )
            # 小步长二次精修，避免停在粗单纯形跨不过的窄谷
            x, val, ok2 = _nelder_mead(
                cost, x, lo, hi, step_frac=0.004
            )
            ok = ok and ok2
            if best is None or val < best[0]:
                best = (val, x, ok)
    assert best is not None
    (_, (f, lam), converged) = best
    return _build_single_result(segments, float(f), float(lam), converged)


def _build_single_result(
    segments: list[Segment], f: float, lam: float, converged: bool
) -> ModelResult:
    lin = solve_linear(segments, np.array([f]), lam)
    res = ModelResult(
        kind="single",
        frequencies=[f],
        decay_rate=lam,
        tau_ms=1000.0 / lam,
        converged=converged,
    )
    for seg, lf in zip(segments, lin):
        a, b = float(lf.coeffs[0]), float(lf.coeffs[1])
        amp = float(np.hypot(a, b))
        phase = float(np.arctan2(-b, a))  # x = A·cos(2πft+φ)，φ=atan2(-b,a)
        res.fit_residual_ratio.append(
            lf.residual_energy / seg.fit_energy
        )
        res.holdout_ratio.append(lf.holdout_error_ratio)
        res.components.append({
            "segment": seg.index,
            "amplitude": amp,
            "phase_rad": phase,
            "a": a,
            "b": b,
        })
        res.traces.append(_trace(seg, lf))
    return res


# ---- 双频拟合 --------------------------------------------------------------
def fit_dual(
    segments: list[Segment],
    f_lo: float,
    f_hi: float,
    single: ModelResult | None = None,
) -> ModelResult:
    # 内部编码 x = [中心频率, 半间隔, λ]：
    # center∈[f_lo, f_hi]、half_gap∈[0, (f_hi-f_lo)/2]、λ∈[λ_min, λ_max]
    half_gap_max = (f_hi - f_lo) / 2.0
    lo = np.array([f_lo, 0.0, LAMBDA_MIN])
    hi = np.array([f_hi, half_gap_max, LAMBDA_MAX])

    def unpack(x: np.ndarray) -> tuple[float, float, float]:
        # 内部以 f1=中心、f2=间距/2 编码，天然保证 f_low < f_high
        center, half_gap, lam = x
        return center - half_gap, center + half_gap, lam

    def cost(x: np.ndarray) -> float:
        f1, f2, lam = unpack(x)
        penalty = _penalty(x, lo, hi)
        if f1 < f_lo or f2 > f_hi or not (LAMBDA_MIN <= lam <= LAMBDA_MAX):
            return penalty
        if f2 - f1 < FREQ_GAP_HZ:
            penalty += 1e6 * (FREQ_GAP_HZ - (f2 - f1)) ** 2
        return _total_residual(
            segments, np.array([f1, f2]), lam
        ) + penalty

    # 候选对：各段 FFT 峰两两组合（间隔 ≥5 Hz），再补「最强峰 + 单频残差峰」
    f_candidates = fft_peak_candidates(segments, f_lo, f_hi, top_k=5)
    if single is None:
        single = fit_single(segments, f_lo, f_hi)
    f0_main = single.frequencies[0]
    residual_peaks = _residual_peaks(segments, single, f_lo, f_hi)
    for rp in residual_peaks[:2]:
        if all(abs(rp - c) > 1.0 for c in f_candidates):
            f_candidates.append(rp)

    starts: list[np.ndarray] = []

    def add_start(fl: float, fh: float, lam0: float = 60.0) -> None:
        fl, fh = sorted((fl, fh))
        x = np.array([(fl + fh) / 2.0, (fh - fl) / 2.0, lam0])
        if all(np.max(np.abs(x - s)) > 1e-6 for s in starts):
            starts.append(x)

    for i, fa in enumerate(f_candidates):
        for fb in f_candidates[i + 1:]:
            if abs(fb - fa) >= FREQ_GAP_HZ:
                add_start(fa, fb)
    for rp in residual_peaks[:2]:
        if abs(rp - f0_main) >= FREQ_GAP_HZ:
            add_start(f0_main, rp)
    # 近频网格：FFT 分辨率不足以分开近频双峰时，用主频率附近的 5~40Hz
    # 细网格补起点，保证 5Hz 间隔的双频也有机会被搜到
    for gap in (5.0, 8.0, 15.0, 30.0):
        for anchor in [f0_main, *f_candidates[:2]]:
            if anchor - gap >= f_lo:
                add_start(anchor - gap, anchor)
            if anchor + gap <= f_hi:
                add_start(anchor, anchor + gap)
    if not starts:
        # 频带内没有间隔 5 Hz 的两个峰：直接构造一个覆盖全带的起点，
        # 由优化器与后续 5 Hz 规则决定能否成立
        add_start(f_lo + 0.25 * (f_hi - f_lo),
                  f_lo + 0.75 * (f_hi - f_lo))

    # 预筛：先在每个起点单点评估代价，只对最优的若干起点做完整
    # Nelder-Mead，避免在明显劣质的起点上重复长迭代。
    # 不能只留极少数：正确盆地的起点可能因一段数据不支持而初始代价偏高
    ranked = sorted(starts, key=cost)[:9]
    best = None
    for x0 in ranked:
        # 两阶段：大步长粗搜定位盆地，小步长精修钻窄谷
        x, val, ok = _nelder_mead(cost, x0, lo, hi, step_frac=0.02)
        x, val, ok2 = _nelder_mead(cost, x, lo, hi, step_frac=0.003)
        ok = ok and ok2
        if best is None or val < best[0]:
            best = (val, x, ok)
    assert best is not None
    (_, x, converged) = best
    f1, f2, lam = unpack(x)
    f1, f2 = float(f1), float(f2)
    if f1 > f2:
        f1, f2 = f2, f1
    return _build_dual_result(segments, [f1, f2], float(lam), converged)


def _residual_peaks(
    segments: list[Segment], single: ModelResult,
    f_lo: float, f_hi: float,
) -> list[float]:
    """单频拟合残差再做 FFT，找最强峰外的候选频率（双频的第二分量线索）。"""
    n_target = max(len(s.fit) for s in segments)
    nfft = 1
    while nfft < n_target * 8:
        nfft *= 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SAMPLE_RATE)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    band_idx = np.where(band)[0]
    found: list[tuple[float, float]] = []
    f_main = single.frequencies[0]
    for seg, tr in zip(segments, single.traces):
        residual = seg.fit - tr["pred_fit_full"]
        window = np.hanning(len(residual))
        spectrum = np.abs(np.fft.rfft(residual * window, n=nfft))
        spec_band = spectrum[band_idx]
        p = int(np.argmax(spec_band))
        gi = band_idx[p]
        y0, y1, y2 = spectrum[gi - 1], spectrum[gi], spectrum[gi + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
        f = float(np.clip(
            freqs[gi] + delta * (freqs[1] - freqs[0]), f_lo, f_hi
        ))
        if abs(f - f_main) >= FREQ_GAP_HZ:
            found.append((float(spec_band[p]), f))
    found.sort(reverse=True)
    return [f for _, f in found]


def _build_dual_result(
    segments: list[Segment],
    freqs: list[float],
    lam: float,
    converged: bool,
) -> ModelResult:
    lin = solve_linear(segments, np.array(freqs), lam)
    res = ModelResult(
        kind="dual",
        frequencies=list(freqs),
        decay_rate=lam,
        tau_ms=1000.0 / lam,
        converged=converged,
    )
    for seg, lf in zip(segments, lin):
        for k, f in enumerate(freqs):
            a, b = float(lf.coeffs[2 * k]), float(lf.coeffs[2 * k + 1])
            res.components.append({
                "segment": seg.index,
                "frequency": f,
                "amplitude": float(np.hypot(a, b)),
                "phase_rad": float(np.arctan2(-b, a)),
                "a": a,
                "b": b,
            })
        res.fit_residual_ratio.append(
            lf.residual_energy / seg.fit_energy
        )
        res.holdout_ratio.append(lf.holdout_error_ratio)
        res.traces.append(_trace(seg, lf))
    res.ablation = _ablation(segments, freqs, lam)
    return res


def _component_prediction(
    t: np.ndarray, f: float, lam: float, a: float, b: float
) -> np.ndarray:
    return np.exp(-lam * t) * (
        a * np.cos(2.0 * np.pi * f * t) + b * np.sin(2.0 * np.pi * f * t)
    )


def _ablation(
    segments: list[Segment], freqs: list[float], lam: float
) -> list[dict]:
    """单独去掉任一分量后，重算两段 holdout 误差占比及增量。"""
    lin = solve_linear(segments, np.array(freqs), lam)
    out: list[dict] = []
    for drop in range(len(freqs)):
        keep = [k for k in range(len(freqs)) if k != drop]
        per_seg = []
        for seg, lf in zip(segments, lin):
            pred = np.zeros_like(seg.t_holdout)
            for k in keep:
                pred += _component_prediction(
                    seg.t_holdout, freqs[k], lam,
                    float(lf.coeffs[2 * k]), float(lf.coeffs[2 * k + 1]),
                )
            err = float(np.dot(seg.holdout - pred, seg.holdout - pred))
            per_seg.append(err / seg.holdout_energy)
        out.append({
            "dropped_frequency": freqs[drop],
            "holdout_ratio": per_seg,
            "delta": [
                per_seg[s] - lin[s].holdout_error_ratio
                for s in range(len(segments))
            ],
        })
    return out


# ---- 波形轨迹（供前端画图，等距降采样到 ≤512 点）--------------------------
def _trace(seg: Segment, lf: LinearFit) -> dict:
    def downsample(x: np.ndarray) -> np.ndarray:
        if len(x) <= 512:
            return x.astype(np.float64)
        bins = np.array_split(x, 512)
        return np.array([float(np.mean(b)) for b in bins])

    actual_full = np.concatenate([seg.fit, seg.holdout])
    pred_full = np.concatenate([lf.pred_fit, lf.pred_holdout])
    return {
        "segment": seg.index,
        "start_sample": seg.start_sample,
        "n_samples": len(seg.samples),
        "fit_split_sample": len(seg.fit),
        "actual": downsample(actual_full).tolist(),
        "prediction": downsample(pred_full).tolist(),
        "pred_fit_full": lf.pred_fit,   # 残差峰计算用，序列化前由 service 删除
    }


# ---- 顶层分析与判定 --------------------------------------------------------
def analyze(
    samples: np.ndarray,
    ranges: list[tuple[int, int]],
    f_lo: float,
    f_hi: float,
) -> dict:
    """完整分析流程，返回可直接序列化的结论字典。"""
    f_lo, f_hi = validate_band(f_lo, f_hi)
    segments = make_segments(samples, ranges)
    ordered_ranges = sorted(ranges, key=lambda r: r[0])

    single = fit_single(segments, f_lo, f_hi)
    single_pass = all(r <= HOLDOUT_LIMIT for r in single.holdout_ratio)

    result: dict = {
        "band_hz": [f_lo, f_hi],
        "segments": [
            {
                "index": s.index,
                "start_sample": ordered_ranges[s.index][0],
                "end_sample": ordered_ranges[s.index][1],
                "duration_ms": round(s.duration_ms, 2),
                "fit_samples": len(s.fit),
                "holdout_samples": len(s.holdout),
            }
            for s in segments
        ],
        "single": _public_model(single),
        "dual": None,
        "status": None,
        "frequencies_hz": None,
        "decay_rate": None,
        "tau_ms": None,
        "holdout_ratios": None,
        "reasons": [],
        "warnings": [],
    }
    if not single.converged:
        result["warnings"].append("单频优化未在限定迭代内完全收敛，结果仅供参考")

    if single_pass:
        result["status"] = "single"
        result["frequencies_hz"] = [round(f, 2) for f in single.frequencies]
        result["decay_rate"] = round(single.decay_rate, 3)
        result["tau_ms"] = round(single.tau_ms, 2)
        result["holdout_ratios"] = [
            round(r, 4) for r in single.holdout_ratio
        ]
        return result

    # 单频不通过 -> 双频（复用已完成的单频结果，避免重复计算）
    dual = fit_dual(segments, f_lo, f_hi, single=single)
    result["dual"] = _public_model(dual)
    if not dual.converged:
        result["warnings"].append("双频优化未在限定迭代内完全收敛，结果仅供参考")

    reasons: list[str] = []
    holdout_ok = all(r <= HOLDOUT_LIMIT for r in dual.holdout_ratio)
    gap = abs(dual.frequencies[1] - dual.frequencies[0])
    gap_ok = gap >= FREQ_GAP_HZ - 1e-9
    ablation_ok = all(
        all(d >= ABLATION_DELTA - 1e-12 for d in ab["delta"])
        for ab in dual.ablation
    )

    if not holdout_ok:
        bad = [str(i + 1) for i, r in enumerate(dual.holdout_ratio)
               if r > HOLDOUT_LIMIT]
        reasons.append(
            "双频模型第 " + "、".join(bad) +
            " 段 holdout 预测误差能量占比超过 10%，预测无法外推到尾音后 1/3"
        )
    if not gap_ok:
        reasons.append(
            f"两个频率仅相隔 {gap:.2f} Hz，不足 {FREQ_GAP_HZ:.0f} Hz，"
            "无法判为两个独立共振"
        )
    if not ablation_ok:
        for ab in dual.ablation:
            weak_segs = [str(i + 1) for i, d in enumerate(ab["delta"])
                         if d < ABLATION_DELTA - 1e-12]
            if weak_segs:
                reasons.append(
                    f"去掉 {ab['dropped_frequency']:.1f} Hz 分量后，第 "
                    + "、".join(weak_segs) +
                    " 段误差占比增加不足 10 个百分点，该分量缺乏独立贡献"
                )

    if holdout_ok and gap_ok and ablation_ok:
        result["status"] = "dual"
        result["frequencies_hz"] = [round(f, 2) for f in dual.frequencies]
        result["decay_rate"] = round(dual.decay_rate, 3)
        result["tau_ms"] = round(dual.tau_ms, 2)
        result["holdout_ratios"] = [
            round(r, 4) for r in dual.holdout_ratio
        ]
        return result

    result["status"] = "unconfirmed"
    result["reasons"] = reasons
    return result


def _public_model(model: ModelResult) -> dict:
    """把模型结果转成可 JSON 序列化的字典，并去掉内部字段。"""
    traces = []
    for tr in model.traces:
        traces.append({k: v for k, v in tr.items() if k != "pred_fit_full"})
    return {
        "kind": model.kind,
        "frequencies_hz": [round(f, 2) for f in model.frequencies],
        "decay_rate": round(model.decay_rate, 3),
        "tau_ms": round(model.tau_ms, 2),
        "converged": model.converged,
        "fit_residual_ratio": [round(r, 4) for r in model.fit_residual_ratio],
        "holdout_ratio": [round(r, 4) for r in model.holdout_ratio],
        "components": [
            {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in c.items()
            }
            for c in model.components
        ],
        "ablation": [
            {
                "dropped_frequency_hz": round(ab["dropped_frequency"], 2),
                "holdout_ratio": [round(r, 4) for r in ab["holdout_ratio"]],
                "delta": [round(d, 4) for d in ab["delta"]],
            }
            for ab in model.ablation
        ],
        "traces": traces,
    }
