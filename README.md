# 模具敲击尾音共振核对

检修员停机敲击模具后，在核对页上传敲击录音，圈出**同一模具两次敲击后的尾音**并填入待查频带，系统用衰减正弦模型拟合尾音，确认后续该查的共振频率。

## 输入约定

- WAV（PCM）：16 kHz、16-bit、单声道，单个文件不超过 8 秒
- 两段尾音：每段 200～500 毫秒，两段互不重叠（端点相接允许）
- 待查频带：上下限均在 100～2000 Hz 内，下限小于上限

## 模型与判定规则

尾音模型（衰减正弦叠加，t 从各段起点计）：

```
单频  x(t) = exp(-λt)·(a·cos(2πft) + b·sin(2πft))
双频  x(t) = Σ_k exp(-λt)·(a_k·cos(2πf_k t) + b_k·sin(2πf_k t))
```

1. **只拟合前 2/3**：每段前 2/3 样本参与拟合（fit 区），后 1/3 为 holdout 区，只用于预测核对，绝不参与参数估计。
2. **两段共用频率与衰减率**：两次敲击的频率 {f_k}、衰减率 λ 完全相同；每段各自拟合振幅 a、b（即振幅与相位）。
3. **频率限在待查频带内**：非线性优化（Nelder-Mead 单纯形法，仅依赖 numpy）的搜索范围受频带硬约束，起点来自加窗 FFT 峰与近频细网格。
4. **单频判定**：两段 holdout 的预测误差能量占各自实录能量的比例都 **≤10%** 才通过，报单频。
5. **双频判定**：单频不通过才尝试双频，须同时满足
   - 两个频率至少相隔 **5 Hz**；
   - 两段 holdout 误差占比都 **≤10%**；
   - 从预测中单独去掉任一分量后，**两段**误差占比都各增加至少 **10 个百分点**（消融判据，保证两个分量都有独立贡献）。
6. **无法确认**：单频不通过且双频不满足全部判据时，撤下旧结论，页面明确列出失败原因（哪一段误差超标、频率间隔不足、哪个分量缺乏独立贡献）。
7. **无效输入不覆盖结果**：格式错误、段长/重叠/频带不合法、静音段等返回 400 且不写入状态；一次完成的分析（含 unconfirmed）会覆盖旧结论，相当于按最新证据撤下旧判断。

页面输出：确认的共振频率（1 或 2 个）、共用衰减率 λ 与时间常数 τ、两段 holdout 误差占比、单/双频模型明细表（含消融增量）、两段实录与预测波形对照（标注 fit/holdout 分界）。

## 启动（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

录音与结果数据保存在命名卷 `press-data`（容器内 `/data`）。双频分析计算量较大，典型耗时 10～20 秒，页面会显示「分析中」。

## 本地开发（无 Docker 时）

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 自定义数据目录：APP_DATA_DIR=/some/dir uvicorn app.main:app
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/analyze` | multipart 表单：`wav`（.wav）、`seg1_start/seg1_end/seg2_start/seg2_end`（采样点整数）、`band_lo/band_hi`（Hz），返回核对结果 |
| `GET` | `/api/result/last` | 最近一次分析结果（无结果时 404） |
| `DELETE` | `/api/result/last` | 清除内存中的最近结果（撤下结论） |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/data/uploads/{name}` | 回传已上传录音，供网页播放器复听 |

结果关键字段：`status`（`single` / `dual` / `unconfirmed`）、`frequencies_hz`、`decay_rate`（λ，1/s）、`tau_ms`（τ=1/λ）、`holdout_ratios`（两段误差占比）、`single`/`dual`（两个模型的完整明细，`dual.ablation` 为消融结果）、`reasons`（不可确认的原因）、`segments`（段采样点位置与 fit/holdout 切分）。

## 测试

```bash
python3 -m pytest tests/ -q
```

覆盖：WAV 格式校验、段长 200～500ms/互不重叠/频带 100～2000Hz 校验、单频恢复与 10% 误差门槛、双频全判据（含 5Hz 最小间隔）、消融否决、纯噪声不误报、unconfirmed 覆盖旧结论、无效输入 400 且不覆盖结果、端到端 API 与页面可达。

合成 WAV 生成器（含三个应分别得到 single / dual / unconfirmed 的演示文件）：

```bash
python3 -I scripts/gen_demo.py /tmp/demo
```

## 目录结构

```
app/
  audio.py       WAV 校验与解析（16k/16bit/单声道）
  resonance.py   衰减正弦拟合、单/双频判定、消融、段与频带校验
  service.py     分析服务与结果存储（无效输入不覆盖旧结果）
  main.py        FastAPI 路由
  static/        单页前端（波形拖选圈段、频带填写、结论与波形对照）
tests/           音频、算法、端到端 API 测试（27 项）
scripts/         演示 WAV 生成
```
