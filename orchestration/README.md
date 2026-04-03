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

然后编辑 `.env`，填入真实认证信息（含 `ARK_API_KEY` 与 `US3_*`）。

说明：
- 脚本会通过 `python-dotenv` 自动读取仓库根目录 `.env`。
- `.env` 已被 `.gitignore` 忽略，不会进入版本库。

## 3. 启动调度服务

```bash
pixi run -e orchestrator pipeline-serve
```

默认调度：
- cron: `0 2 1 * *`
- timezone: `Asia/Shanghai`
- 自动执行上个月数据
- 默认数据源：`cpc`（可切换 `psl`）
- `model` 环境仅执行推理；报告分析与 PDF 生成在 `orchestrator` 环境执行
- Prefect 中可见推理阶段：依赖检查、数据下载、预处理、模型推理、结果转换、报告生成
- `cpc` 下载缓存固定目录：`./tmp/cpc_cache`（容器内 `/app/tmp/cpc_cache`）
- 推理临时文件统一在 `./tmp` 下通过 `TemporaryDirectory` 管理并自动清理
- 最终产物上传到：`s3://fengwu-public/szcx_ocean_report/YYYY-MM.pdf`
- 本地 `output` 不保留当前流程生成的 PDF 文件
- 报告模板固定来源：`orchestration/reporting/assets/report_template.md`
- 报告分析脚本固定来源：`orchestration/reporting/assets/analyzer.py`

## 4. 手动运行

先做 dry-run：

```bash
pixi run -e orchestrator pipeline-trigger -- --param dry_run=true
```

实际执行：

```bash
pixi run -e orchestrator pipeline-trigger -- --param dry_run=false
```

指定月份：

```bash
pixi run -e orchestrator pipeline-trigger -- --param target_month=2026-02 --param dry_run=false
```

指定数据源（例如 PSL）：

```bash
pixi run -e orchestrator pipeline-trigger -- --param source=psl --param target_month=2026-02 --param dry_run=false
```

注意：`target_month` 必须是 `YYYY-MM`（如 `2026-02`）。
注意：`source` 仅支持 `cpc` / `psl`，默认 `cpc`。
注意：`pipeline-trigger` 不再自动读取 `.env`，请确保当前运行环境已提供 `PREFECT_API_URL` 与 `PREFECT_API_AUTH_STRING`。

## 5. 自动部署（GitHub Actions）

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
- Secrets: `UHUB_USERNAME`, `UHUB_PASSWORD`, `DEPLOY_SSH_KEY`, `PREFECT_API_AUTH_STRING`, `ARK_API_KEY`, `US3_PUBLIC_KEY`, `US3_PRIVATE_KEY`
- Vars: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PATH`, `PREFECT_API_URL`, `US3_END_POINT`

部署前置：
- 服务器需预置模型与统计文件（不随镜像分发）：
  - `${DEPLOY_PATH}/ckpt/seed_1.bin`
  - `${DEPLOY_PATH}/stat/mean/*` 与 `${DEPLOY_PATH}/stat/std/*`
