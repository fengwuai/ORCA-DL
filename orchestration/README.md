# Prefect 月度推理流水线

本目录提供 ORCA-DL 的 Prefect 自动调度。
流程会在推理完成后自动生成 PDF 报告。

任务与部署名称固定为：`海洋模型预测`。

## 1. 环境准备

```bash
pixi install -e orchestrator
pixi install -e model
```

## 2. 配置 `.env`

```bash
cp .env.example .env
```

然后编辑 `.env`，填入真实认证信息（含 `ARK_API_KEY` 与 `XIAMEN_S3_*`）。

说明：
- 脚本会通过 `python-dotenv` 自动读取仓库根目录 `.env`。
- `.env` 已被 `.gitignore` 忽略，不会进入版本库。

## 3. CLI 命令（仅保留两条业务命令）

### 3.1 完整流程（推理 -> 生成报告 -> 上传）

```bash
pixi run -e orchestrator pipeline
pixi run -e orchestrator pipeline 2026-02 --source psl --output-dir ./output/reports
```

参数：
- `target_month`（可选）：格式 `YYYY-MM`，默认上个月（`Asia/Shanghai`）
- `--output-dir`：报告输出目录，默认 `./output/reports`
- `--source`：初始数据源，`cpc`（默认）或 `psl`

行为说明：
- 推理中间目录、模型中间结果、临时 NetCDF 均由 `TemporaryDirectory` 统一管理并自动清理
- 最终上传：`s3://szcx-ds-wthr-public/ocean_report/YYYY-MM.pdf`
- 本地保留 PDF：`{output_dir}/YYYY-MM.pdf`

### 3.2 仅推理流程

```bash
pixi run -e model inference
pixi run -e model inference 2026-02 --source cpc --output-dir ./output/models
```

参数：
- `target_month`（可选）：格式 `YYYY-MM`，默认上个月（`Asia/Shanghai`）
- `--output-dir`：模型输出目录，默认 `./output/models`
- `--source`：初始数据源，`cpc`（默认）或 `psl`

行为说明：
- 输出 NetCDF：`{output_dir}/YYYY-MM.nc`
- 下载/预处理/推理中间文件统一在 `./tmp` 下临时管理并自动清理

## 4. Prefect 调度

启动部署服务：

```bash
pixi run -e orchestrator pipeline-serve
```

默认调度：
- cron: `0 2 20 * *`
- timezone: `Asia/Shanghai`
- 自动执行上个月数据
- Prefect Flow 仅调用单一 `pipeline` 函数（不再做 stage 级编排）
- 默认数据源：`cpc`（可切换 `psl`）
- 报告模板固定来源：`orchestration/reporting/assets/report_template.md`
- 报告分析脚本固定来源：`orchestration/reporting/assets/analyzer.py`

## 5. 厦门上传模块

新增独立模块：`orchestration/xiamen_uploader.py`。

说明：
- 只读取两个环境变量：`XIAMEN_S3_AK`、`XIAMEN_S3_SK`
- 其他参数（bucket/region/internal ip/public domain/path）在模块中固定
- 上传路径前缀固定为：`ocean_report/`
- 重试逻辑使用 `tenacity`

调用示例：

```python
from orchestration.xiamen_uploader import upload_file_to_xiamen

uri = upload_file_to_xiamen("./output/reports/2026-02.pdf")
# 或指定对象名（仍上传到 ocean_report/ 前缀下）
uri = upload_file_to_xiamen("./output/reports/2026-02.pdf", object_name="custom-name.pdf")
print(uri)
```

## 6. 自动部署（GitHub Actions）

仓库内已提供部署资产：
- `deploy/Dockerfile`
- `deploy/docker-compose.prod.yml`
- `.github/workflows/deploy.yml`

发布策略：
- `push main` 自动构建并发布镜像，再 SSH 到服务器执行 `docker compose up -d`。
- 工作流固定镜像仓库参数：
  - `REGISTRY=uhub.usuanova.com`
  - `IMAGE_NAME=xiangfeng/orca-dl-pipeline`

需要配置的 GitHub Secrets / Vars（命名与当前部署工作流一致）：
- Secrets: `UHUB_USERNAME`, `UHUB_PASSWORD`, `DEPLOY_SSH_KEY`, `PREFECT_API_AUTH_STRING`, `ARK_API_KEY`, `XIAMEN_S3_AK`, `XIAMEN_S3_SK`
- Vars: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PATH`, `PREFECT_API_URL`

部署前置：
- 服务器需预置模型与统计文件（不随镜像分发）：
  - `${DEPLOY_PATH}/ckpt/seed_1.bin`
  - `${DEPLOY_PATH}/stat/mean/*` 与 `${DEPLOY_PATH}/stat/std/*`
