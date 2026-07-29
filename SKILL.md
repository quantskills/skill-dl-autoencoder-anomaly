---
name: skill-dl-autoencoder-anomaly
description: 深度自编码器无监督异常检测。每日盘后对沪深300成分股用最近60日窗口重训一个MLP自编码器，输出T日重建误差Top-10异常股（CSV + Markdown）。
tags: [quant, anomaly, autoencoder, deep-learning, csi300]
---

# 深度自编码器 · 沪深300 无监督异常检测

## 适用场景
- 每日盘后想快速看"今天沪深300里哪几只股票的走势最不像自己往常的样子"
- 想用无监督方式挖掘"看不出具体理由但市场行为异常"的个股
- 想跟踪某个日期的沪深300横截面中最偏离平均画像的异常股

## 数据接口（panda_data）

| 接口 | 用途 | 关键字段 |
|---|---|---|
| `get_last_trade_date` | 解析最近交易日 | `date` |
| `get_prev_trade_date` | 计算 T-60 训练窗口起点 | `date` |
| `get_index_weights` | 锁定 T 日 CSI300 成分 | `index_symbol, date, stock_symbol, weight` |
| `get_factor` | 一次拉取 OHLCV + turnover + market_cap | `date, symbol, name, open, close, high, low, volume, amount, turnover, market_cap` |
| `get_stock_daily_post` | 后复权 OHLCV + 涨跌停价 + 停牌标记 | `date, symbol, pre_close, limit_up, limit_down, trade_status` |
| `get_index_daily` | 拉取基准指数走势用于超额收益 | `symbol, date, close, pre_close` |

字段详见 `references/need_used_api.md`。

## 术语约定

- `reconstruction_error` = MSE(x, AE(x))，值越大越异常
- `top_feature` = per-feature-column MSE 里贡献最大的原始特征名（帮助解释异常方向）
- **训练集严格不含 T 日**：`date < T` 显式过滤，保证无未来函数

首次实测须校准列名与 `get_factor` 实际返回；若字段缺失或改名，编辑 `scripts/data.py` 的 `EXPECTED_COLUMNS`。

## 股票池（Universe）

**沪深300 成分股**（默认 `--index 000300.SH`）：

- 取 T 日 `get_index_weights(index_symbol=--index, start_date=T, end_date=T)` 的 `stock_symbol` 列
- 若 T 日无权重发布（周末/节假日效应），回退到最近发布日的成分
- CSI300 天然是流动性最好的 300 只非 ST 股票，无需再叠加 ST/停牌/新股过滤（停牌日会在训练/打分阶段按 `trade_status` 剔除）

未来可切换 `--index 000905.SH`（中证500）或 `--index 000852.SH`（中证1000）。

## 数据回看窗口 · 训练 60 交易日 + 特征 20 交易日

- 训练集覆盖 `[T-60, T-1]`（60 个交易日），每只股票在每一天有一个 20 日窗口样本
- 20 日窗口用 `get_prev_trade_date(date=T, n=80)` 求下界（60 训练日 + 20 特征日）
- 严格按交易日回退，不引入自然日 buffer

## 特征工程

对每只股票、每个交易日，构造 **8 维日频特征**：

| # | 名称 | 公式 | 说明 |
|---|------|------|------|
| 1 | `ret` | `close / pre_close - 1` | 日收益 |
| 2 | `log_vol` | `log(volume + 1)` | 对数成交量 |
| 3 | `amplitude` | `(high - low) / pre_close` | 振幅 |
| 4 | `turnover` | `turnover` | 换手率 |
| 5 | `gap` | `open / pre_close - 1` | 开盘跳空 |
| 6 | `dist_limit_up` | `(limit_up - close) / close` | 距涨停 |
| 7 | `dist_limit_down` | `(close - limit_down) / close` | 距跌停 |
| 8 | `excess_ret` | `ret - index_ret` | 相对 CSI300 超额收益 |

- **样本 = 20 日 × 8 特征展平为 160 维向量**
- **标准化**：训练集上按列（跨股票 × 跨时间）算 `mean/std`，z-score 标准化；同样的 mean/std 应用于 T 日打分样本（防泄露）
- **缺失处理**：训练窗口内 `trade_status != 0` 或 `pre_close` 缺失日 > 5 的股票整只剔除；T 日打分窗口有任意日缺失则不出现在榜单

## 模型

**MLP Autoencoder**：`160 → 96 → 48 → 32 → 48 → 96 → 160`

- 每层 `Linear + ReLU + Dropout(0.1)`，decoder 最后一层不加 ReLU
- 参数量 ≈ 45k，CPU 上 30 秒到 2 分钟可训完
- 损失：MSE (逐元素均方误差)
- 优化器：Adam，`lr=1e-3`
- Batch size 256，最多 50 epochs，early stopping patience=5（验证集 MSE 停降 5 轮就停）
- 训练集/验证集 80/20 随机划分
- 随机种子固定 `--seed 42`，`torch.manual_seed + numpy.random.seed`

## 输出结果

**`output/anomaly_YYYYMMDD.csv`**（Top-N 每只股票一行）：

| 列 | 说明 |
|---|---|
| `trade_date` | 扫描日 T |
| `symbol` | 股票代码 |
| `name` | 股票名 |
| `rank` | 榜单排名（1 = 最异常） |
| `reconstruction_error` | 重建 MSE |
| `top_feature` | 主导异常的原始特征名 |
| `ret_T` | T 日收益 |
| `turnover_T` | T 日换手率 |
| `amplitude_T` | T 日振幅 |
| `detail_json` | `{per_feature_mse: {...}, z_score: <float>, trained_samples: <int>}` |

**排序**：按 `reconstruction_error` 降序，取 Top-N（默认 10）。

**`output/anomaly_YYYYMMDD.md`**：Top-N 表 + 训练元信息 + 一句解读。

## 使用方式

```bash
# 认证 & 环境（首次）
conda activate pandaai
pip install -r requirements.txt   # 注意 torch 会拉取 ~200MB
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 字段自检（首次使用 / panda_data 版本更新后跑一次）
python -m scripts.data --self-check --date 20260729

# 单日扫描 —— 默认最近交易日，默认 CSI300
python scripts/scan.py

# 指定日期
python scripts/scan.py --date 20260729

# 完整参数
python scripts/scan.py \
    --date 20260729 \
    --index 000300.SH \
    --lookback 20 \
    --train_days 60 \
    --top_n 10 \
    --epochs 50 \
    --batch_size 256 \
    --seed 42

# 单元测试
pytest tests/ -v
```

## 验收要求
- **无未来函数**：训练集严格 `date < T`，`test_features.py::test_no_lookahead` 覆盖
- **单元测试全通过**：`pytest tests/` 无失败，总数 ≥ 20
- **字段自检通过**：`python -m scripts.data --self-check --date <近期日>` 返回 0
- **端到端跑通**：至少一个真实日期能产出 CSV + MD，Top-10 完整无 NaN
- **训练可复现**：相同 `--seed` 两次运行的 Top-10 顺序完全一致
- **文档一致**：本文件的特征公式与 `scripts/features.py` 一致

## 已知局限
- Universe 固定 CSI300（v0.2 支持 CSI500/CSI1000 + 按 L1 行业分组训练）
- 每次运行都重训，不做 checkpoint 复用（v0.2 支持模型缓存）
- 只输出重建误差，不融合龙虎榜 / 涨跌停 / 大宗交易等外部"官方异常"信号（v0.2 增加对比表）
- 单一 MLP 全市场；v0.2 增加按 L1 行业训 31 个子模型的选项
- `top_feature` 只回归到 8 个原始特征名，不细分到 20 日中的哪一天（v0.2 增加时序热力图）
- 依赖 `get_factor` 的 `turnover` 与 `market_cap` 字段完整性；首次校准时若缺列，自动回退到 `get_stock_daily_post`（需要动手编辑 `scripts/data.py` 选择数据源）
- **首次实测发现**：`get_factor` 实际不返回 `name` 列（文档写有但响应无），已从 `get_stock_daily_post` 补入。
- **复权跳变未过滤**：若一段窗口内某股票经历除权除息或送股等极端调整，其 `ret`/`amplitude` 会出现异常大的数值（20260729 榜单 Top-2 国电电力 ret_T=318 即此类情形）。v0.2 计划在 `features.py` 增加 `|ret| < 0.11` 的软过滤，剔除明显的复权跳变样本。
