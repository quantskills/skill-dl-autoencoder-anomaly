# skill-dl-autoencoder-anomaly · Design (2026-07-30)

## 1. 目的

在 `/Users/since/Code/quantskills/` 下新建一个基于**深度自编码器 (MLP AE) 的沪深300 无监督异常检测** skill。每晚一条命令,对 CSI300 每只成分股基于近 60 日多维特征进行 AE 重建,输出重建误差 Top-N 的异常股票榜单 (CSV + Markdown)。

## 2. 关键决策 (brainstorming)

| 决策项 | 结论 |
|---|---|
| 异常检测范围 | 横截面 + 时序混合,universe = CSI 300 (`000300.SH`) |
| 模型架构 | 深 MLP Autoencoder (`160 → 96 → 48 → 32 → 48 → 96 → 160`) |
| 训练策略 | 每次运行都用 T-60 到 T-1 的短窗口重训,不落盘 checkpoint |
| 输出量 | Top-10 异常股,按重建误差 desc |
| 外部信号融合 | v0.1 不融合龙虎榜/涨跌停,只输出 AE 重建误差 |
| 运行环境 | conda env `pandaai` + `pip install torch` |

## 3. 数据接口

从 `panda_data_api_doc.md` 抽取 6 个 API:

1. `get_last_trade_date` — 定位最近交易日
2. `get_prev_trade_date` — 计算 T-60 与 T-80 起点
3. `get_index_weights` — CSI300 成分股锁定
4. `get_factor` — OHLCV + turnover + market_cap 一次取全
5. `get_stock_daily_post` — 后复权 OHLCV + limit_up/down + trade_status (补 `get_factor` 缺项)
6. `get_index_daily` — CSI300 基准收益,计算超额收益特征

## 4. 特征

- 8 维日频特征 × 20 日窗口 = 160 维向量
- 训练样本: 300 只 × 40 天 ≈ 12,000 条 (60 日训练窗口去掉前 20 日热身)
- 打分样本: T 日 300 条
- 特征列: `ret, log_vol, amplitude, turnover, gap, dist_limit_up, dist_limit_down, excess_ret`
- z-score 标准化: 训练集算 mean/std,同参数应用到打分样本

## 5. 无未来函数

- 训练集严格 `date < T`,字符串比较过滤
- 标准化参数 `mean/std` 只在训练集上估计
- 停牌日按 `trade_status != 0` 剔除

## 6. 输出契约

- `output/anomaly_YYYYMMDD.csv` — 10 列: `trade_date, symbol, name, rank, reconstruction_error, top_feature, ret_T, turnover_T, amplitude_T, detail_json`
- `output/anomaly_YYYYMMDD.md` — Top-10 表 + 训练元信息 + 一句解读

## 7. CLI 退出码

- `0` 成功
- `1` panda_data 错误 / 网络
- `2` 指定日期无数据
- `3` CSI300 成分股为空
- `4` 列自检失败

## 8. 验收

见 `SKILL.md ## 验收要求`。

## 9. 已知局限

见 `SKILL.md ## 已知局限`。
