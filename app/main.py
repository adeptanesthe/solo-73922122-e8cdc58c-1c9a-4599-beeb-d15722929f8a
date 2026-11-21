"""FastAPI 入口：上传敲击 WAV 与两段尾音区间、待查频带，返回共振核对结果。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .audio import AudioValidationError, MAX_DURATION_SEC, SAMPLE_RATE
from .resonance import ResonanceError
from .service import MAX_UPLOAD_BYTES, Store

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/workspace/data"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="模具敲击尾音共振核对", version="2.0.0")
store = Store(DATA_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/uploads/{name}")
def get_upload(name: str) -> FileResponse:
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = store.uploads_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/result/last")
def last_result() -> dict:
    result = store.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效检查结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


def _parse_sample(value: str, label: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"{label}必须是采样点整数"
        ) from exc
    if n < 0:
        raise HTTPException(status_code=400, detail=f"{label}不能为负数")
    return n


@app.post("/api/analyze")
async def analyze(
    wav: UploadFile = File(..., description="敲击录音 WAV"),
    seg1_start: int = Form(...),
    seg1_end: int = Form(...),
    seg2_start: int = Form(...),
    seg2_end: int = Form(...),
    band_lo: float = Form(...),
    band_hi: float = Form(...),
) -> dict:
    if not (wav.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="只接受 .wav 文件")
    raw = await wav.read()
    if not raw:
        raise HTTPException(status_code=400, detail="WAV 文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )

    seg1 = (_parse_sample(str(seg1_start), "第 1 段起点"),
            _parse_sample(str(seg1_end), "第 1 段终点"))
    seg2 = (_parse_sample(str(seg2_start), "第 2 段起点"),
            _parse_sample(str(seg2_end), "第 2 段终点"))

    try:
        # 拟合是 CPU 密集型（双频约 10~20s），放入线程池避免阻塞事件循环
        return await run_in_threadpool(
            store.analyze, raw, seg1, seg2, float(band_lo), float(band_hi)
        )
    except (AudioValidationError, ResonanceError) as exc:
        # 任何校验/拟合前错误都在 store 写入前抛出，不会覆盖上一次有效结果
        raise HTTPException(status_code=400, detail=str(exc)) from exc
