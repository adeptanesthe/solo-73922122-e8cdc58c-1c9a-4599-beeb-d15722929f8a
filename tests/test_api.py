"""端到端 API 关键流程测试。"""
import io
import wave

import numpy as np
import pytest

from conftest import SR, build_tail_wav, post_analyze, silent_wav

F1, F2, LAM, DUR = 432.0, 715.0, 14.0, 320


def _single():
    return build_tail_wav([
        {"start_ms": 200, "dur_ms": DUR,
         "components": [(F1, 0.65, 0.3)], "decay_rate": LAM},
        {"start_ms": 1100, "dur_ms": DUR,
         "components": [(F1, 0.5, 1.7)], "decay_rate": LAM},
    ])


def _dual():
    return build_tail_wav([
        {"start_ms": 200, "dur_ms": DUR,
         "components": [(F1, 0.55, 0.2), (F2, 0.35, 1.0)],
         "decay_rate": LAM},
        {"start_ms": 1100, "dur_ms": DUR,
         "components": [(F1, 0.45, 1.8), (F2, 0.4, 2.4)],
         "decay_rate": LAM},
    ])


def test_health_and_empty_result(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/result/last").status_code == 404


def test_single_frequency_flow(client):
    wav, ranges = _single()
    resp = post_analyze(client, wav, ranges)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "single"
    assert body["frequencies_hz"][0] == pytest.approx(F1, abs=1.5)
    assert len(body["holdout_ratios"]) == 2
    assert all(x <= 0.10 for x in body["holdout_ratios"])
    # 结果持久化，可再次获取
    last = client.get("/api/result/last").json()
    assert last["id"] == body["id"]


def test_dual_frequency_flow(client):
    wav, ranges = _dual()
    body = post_analyze(client, wav, ranges).json()
    assert body["status"] == "dual", body.get("reasons")
    assert body["frequencies_hz"][1] - body["frequencies_hz"][0] >= 5.0
    for ab in body["dual"]["ablation"]:
        assert all(d >= 0.10 for d in ab["delta"])


def test_unconfirmed_overwrites_old_conclusion(client):
    # 先得到一个有效的单频结论
    wav_ok, ranges = _single()
    good = post_analyze(client, wav_ok, ranges)
    assert good.status_code == 200
    good_id = good.json()["id"]

    # 再提交纯噪声尾音：结论变为 unconfirmed，旧结论被撤下
    rng = np.random.default_rng(9)
    n = int(3.0 * SR)
    noise = np.clip(rng.standard_normal(n) * 0.05, -1, 1).astype(np.float32)
    pcm = (noise * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    r1 = (int(0.2 * SR), int(0.52 * SR))
    r2 = (int(1.1 * SR), int(1.42 * SR))
    resp = post_analyze(client, buf.getvalue(), [r1, r2])
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unconfirmed"
    assert body["id"] != good_id
    assert body["reasons"]
    stored = client.get("/api/result/last").json()
    assert stored["status"] == "unconfirmed"


def test_invalid_inputs_return_400_and_keep_last_result(client):
    wav, ranges = _single()
    good = post_analyze(client, wav, ranges)
    assert good.status_code == 200
    good_id = good.json()["id"]

    # 段时长不足 200ms
    bad_range = [(int(0.2 * SR), int(0.35 * SR)), ranges[1]]
    resp = post_analyze(client, wav, bad_range)
    assert resp.status_code == 400
    assert "200" in resp.json()["detail"]

    # 两段重叠
    overlap = [ranges[0], (ranges[0][0] + 100, ranges[0][1] + 100)]
    resp = post_analyze(client, wav, overlap)
    assert resp.status_code == 400
    assert "互不重叠" in resp.json()["detail"]

    # 频带超出 100~2000
    resp = post_analyze(client, wav, ranges, band=(50.0, 2000.0))
    assert resp.status_code == 400
    assert "100~2000" in resp.json()["detail"]

    # 非 WAV
    (s1, e1), (s2, e2) = ranges
    resp = client.post(
        "/api/analyze",
        data={"seg1_start": s1, "seg1_end": e1,
              "seg2_start": s2, "seg2_end": e2,
              "band_lo": 100, "band_hi": 2000},
        files={"wav": ("tap.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400

    # 静音段无法拟合
    resp = post_analyze(client, silent_wav(3.0), ranges)
    assert resp.status_code == 400
    assert "静音" in resp.json()["detail"]

    # 上一次有效结果仍保留
    assert client.get("/api/result/last").json()["id"] == good_id


def test_wrong_wav_format_rejected(client):
    wav, ranges = _single()
    # 8kHz 重采样封装
    with wave.open(io.BytesIO(wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    low = io.BytesIO()
    with wave.open(low, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm[::2])
    resp = post_analyze(client, low.getvalue(), ranges)
    assert resp.status_code == 400
    assert "采样率" in resp.json()["detail"]


def test_uploaded_wav_served_back(client):
    wav, ranges = _single()
    body = post_analyze(client, wav, ranges).json()
    resp = client.get(body["audio_url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_clear_result(client):
    wav, ranges = _single()
    post_analyze(client, wav, ranges)
    assert client.delete("/api/result/last").status_code == 200
    assert client.get("/api/result/last").status_code == 404


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "模具" in resp.text
