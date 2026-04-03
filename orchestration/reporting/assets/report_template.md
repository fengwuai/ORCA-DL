# {{report_title}}

**日期**：{{report_date}}  
**数据来源**：{{data_source}}

## 1. 摘要

本报告基于 ORCA-DL 深度学习模型对 {{period_start}} 至 {{period_end}} 的全球海洋状态预测结果，重点分析 **ENSO（厄尔尼诺-南方涛动）** 演变及其对航运业务的潜在影响。

**核心结论**：

- **气候趋势**：{{core_climate_conclusion}}
- **航运影响**：{{core_shipping_conclusion}}

---

## 2. 气候趋势分析：ENSO 演变

我们通过监测 **Nino 3.4 区域** (5°N-5°S, 170°W-120°W) 的海表面温度 (SST) 评估 ENSO 状态。

### 2.1 Nino 3.4 指数时间序列

![Nino 3.4 Time Series](assets/nino34_timeseries.png)

*图 1：Nino 3.4 区域海温预测时间序列，红线为预测期均值。*

### 2.2 阶段划分

根据距平变化，按连续时间窗给出阶段划分：

| 时间段 | 状态 | 距平特征 | 依据 |
| :--- | :--- | :--- | :--- |
| {{phase_window_1}} | {{phase_state_1}} | {{phase_anomaly_1}} | {{phase_basis_1}} |
| {{phase_window_2}} | {{phase_state_2}} | {{phase_anomaly_2}} | {{phase_basis_2}} |

---

## 3. 海洋环境可视化

### 3.1 全球海温分布

以下展示预测初期、中期和末期的全球海温分布：

| {{sst_start_label}} | {{sst_mid_label}} | {{sst_end_label}} |
| :---: | :---: | :---: |
| ![SST Map Start](assets/sst_map_0.png) | ![SST Map Mid](assets/sst_map_12.png) | ![SST Map End](assets/sst_map_23.png) |

*图 2：不同时期全球 SST 分布。*

### 3.2 表面洋流强度

![Mean Current Speed](assets/mean_current_speed.png)

*图 3：预测期内平均表面洋流速度。*

---

## 4. 对航运行业的影响分析

### 4.1 航运风险与建议清单

请按以下固定字段输出编号列表（至少 3 条）：

1. **风险信号**：{{risk_signal_1}}
   - **影响区域/航线**：{{risk_region_1}}
   - **时间窗**：{{risk_window_1}}
   - **运营影响**：{{risk_impact_1}}
   - **建议动作**：{{risk_action_1}}
   - **置信度**：{{risk_confidence_1}}
2. **风险信号**：{{risk_signal_2}}
   - **影响区域/航线**：{{risk_region_2}}
   - **时间窗**：{{risk_window_2}}
   - **运营影响**：{{risk_impact_2}}
   - **建议动作**：{{risk_action_2}}
   - **置信度**：{{risk_confidence_2}}
3. **风险信号**：{{risk_signal_3}}
   - **影响区域/航线**：{{risk_region_3}}
   - **时间窗**：{{risk_window_3}}
   - **运营影响**：{{risk_impact_3}}
   - **建议动作**：{{risk_action_3}}
   - **置信度**：{{risk_confidence_3}}

## 5. 结论

{{final_conclusion}}
