# Prefect 月度推理流水线

本目录提供 ORCA-DL 的 Prefect 自动调度。
流程会在推理完成后自动生成同名 markdown 报告。

任务与部署名称固定为：`海洋模型预测`。

## 1. 环境准备

```bash
pixi install -e orchestrator
```

## 2. 配置 `.env`

```bash
cp .env.example .env
```

然后编辑 `.env`，填入真实认证信息（含 `ARK_API_KEY`）。

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
- 推理完成后生成报告：`output/predictions/orca_dl_prediction_YYYY_MM_24months.md`
- 报告模板固定来源：`demo/20260312000000-自动化海洋报告生成/report.md`

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

注意：`target_month` 必须是 `YYYY-MM`（如 `2026-02`）。
