---
name: skill-dl-autoencoder-anomaly
description: 深度自编码器无监督异常检测 —— 用户问「今天哪些股票走势最异常」「沪深300 异常股」「AE 跑一下」「找不像自己往常样子的股票」类问题时触发。对沪深300成分股用最近60日窗口重训一个MLP自编码器，输出T日重建误差Top-10异常股，按「样式② 结构化播报」呈现。
tags: [quant, anomaly, autoencoder, deep-learning, csi300]
---

# 深度自编码器 · 沪深300 无监督异常检测

## 何时触发本 skill

用户提问命中下列语义时，自动调用：

- 「今天/YYYYMMDD 哪些股票走势最异常」「沪深300 异常股」
- 「跑一下 AE / autoencoder 异常检测」
- 「找不像自己往常样子的股票」「行为异常的个股」
- 「今天有没有偏离画像的股票」

**不触发**：单只股票的多空判断、需要基本面/新闻理由的异常、非 CSI300 池（v0.1 仅支持 CSI300）、涨跌停/龙虎榜等"官方"异常（本 skill 是无监督模型识别的行为异常，不融合官方名单）。

**⚠️ 触发前须知**：本 skill 每次调用会**重新训练一个 MLP AE**（60 交易日窗口，MPS 上 30 epoch 约 30 秒~2 分钟）。Agent 应告知用户"这会训一个模型,约 1 分钟左右",避免用户以为卡死。

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

## Agent 触发流程（本 skill 的正式用法）

用户提问命中「何时触发」后，按五步执行：

### Step 0 · 凭证预检（必须先做，未通过不进入 Step 1）

本 skill 依赖 `panda_data` 服务，需要两个环境变量：

- `PANDA_DATA_USERNAME`
- `PANDA_DATA_PASSWORD`

**Agent 执行前先探测这两个变量**（Bash 是非交互 shell，`~/.zshrc` 里 export 的变量不会自动进来，须显式 source）：

```bash
set -a && source ~/.zshrc >/dev/null 2>&1 && set +a && \
if [ -z "$PANDA_DATA_USERNAME" ] || [ -z "$PANDA_DATA_PASSWORD" ]; then \
    echo "MISSING_CREDENTIALS"; \
else \
    echo "OK: user=$PANDA_DATA_USERNAME"; \
fi
```

**如果输出 `MISSING_CREDENTIALS`**：**立刻停下**，不要继续 Step 1/2/3/4。用下面这段话回复用户（照抄，把 `〈说明〉` 位置替换成实际情况）：

> 跑这个 skill 需要先配置 panda_data 的账号凭证，你还没设置。请按下面两步操作：
>
> 1. 打开 `~/.zshrc`（或 `~/.bashrc`，看你用哪个 shell），在文件末尾加两行：
>    ```bash
>    export PANDA_DATA_USERNAME="你的 panda_data 用户名"
>    export PANDA_DATA_PASSWORD="你的 panda_data 密码"
>    ```
> 2. 保存后，在终端执行 `source ~/.zshrc`（bash 用户执行 `source ~/.bashrc`）让变量生效。
>
> 配置好之后，再问我一次"今天沪深300 哪些股票走势最异常"，我就能跑了。
>
> 如果你还没有 panda_data 账号，需要先在 panda_data 官网注册。

**如果输出 `OK: user=...`**：凭证具备，进入 Step 1。

### Step 1 · 决定扫描日期 + 提前告知耗时

- 用户明说了日期 → 换算为 `YYYYMMDD` 用作 `--date`
- 用户没说 → 省略 `--date`，让 scan 自动取最近交易日
- **调用前先告诉用户**："这会在 MPS/CPU 上训一个 60 日窗口的 AE，约 30 秒~2 分钟，稍等。"

### Step 2 · 调用（推荐一行）

```bash
cd /Users/since/Code/quantskills/skill-dl-autoencoder-anomaly && \
set -a && source ~/.zshrc >/dev/null 2>&1 && set +a && \
/opt/miniconda3/envs/pandaai/bin/python scripts/scan.py [--date YYYYMMDD] --seed 42
```

- 环境是 conda `pandaai`（Python 3.10，`panda_data` + `torch` 已装）
- **`--seed 42` 保持默认**，保证同一日期重复调用结果一致（v0.1.0 复现性已验证）
- 训练不落盘 checkpoint，每次都从头训（v0.2 计划支持缓存）
- exit code：0 OK / 1 panda_data 异常 / 2 该日无数据 / 3 池空 / 4 字段自检失败

### Step 3 · 读取输出

产物固定在两个位置：

- `output/anomaly_YYYYMMDD.csv` —— Top-N 全字段，含 `detail_json` 里的 per-feature MSE
- `output/anomaly_YYYYMMDD.md` —— Top-10 表 + 训练元信息 + 主导特征分布

**直接读 `.md`** 拿排行，需要 per-feature MSE 明细再看 `.csv`。

### Step 4 · 用「样式② 结构化播报」呈现

**不要**把 CSV 路径丢给用户，也**不要**贴 markdown 原文。按固定五段呈现：

```
AE 异常检测 · 沪深300 · YYYYMMDD（参训 N 只，训练窗口 T-60 → T-1）

▎主线判断：<一句话，见下表>

▎最异常 Top 5（按重建误差降序）
- <symbol> <name>：error=X.XXX，主导特征 <feat_name>，T 日收益 +X.XX%，换手 X.X%
- <symbol> <name>：error=X.XXX，主导特征 <feat_name>，...
- <symbol> <name>：error=X.XXX，主导特征 <feat_name>，...
- ...共 5 只

▎主导特征分布（Top 10 里最常见的异常维度）
- <feat>：X/10   <feat>：X/10   <feat>：X/10

▎训练元信息
- 训练样本：N 条 · 验证 MSE：X.XXXX · 设备：mps / cpu · epoch：X

▎异常方向解读（对 Top 3 各写一句）
- <symbol>：<主导特征> 突出，通常意味着 <见特征→现象映射表>
```

**主线判断话术表**：

| 场景 | 话术 |
|---|---|
| Top 10 主导特征集中在 `dist_limit_up` (≥5/10) | 「多股接近涨停,情绪面驱动的异常」 |
| Top 10 主导特征集中在 `turnover` (≥5/10) | 「今日异常主要由换手率放大驱动,资金活跃度突增」 |
| Top 10 主导特征集中在 `amplitude` / `ret` | 「今日异常主要是价格振幅/收益极端,警惕复权跳变污染榜单」 |
| Top 10 主导特征分散(无一个 ≥4/10) | 「异常成因分散,无单一主导模式」 |
| 验证 MSE > 0.30 | 「模型欠拟合(val_MSE 偏高),异常分数需谨慎解读」 |

**特征 → 现象映射表**（Step 4 里"异常方向解读"用这个）：

| 主导特征 | 常见现象 |
|---|---|
| `dist_limit_up` | 距涨停很近或已涨停,情绪推动 |
| `dist_limit_down` | 距跌停很近或已跌停,风险释放 |
| `turnover` | 换手率异常放大,资金进出剧烈 |
| `amplitude` | 日内振幅极大,多空撕扯或消息面 |
| `ret` | 单日收益极端,可能是消息面 or 复权跳变 |
| `log_vol` | 成交量脱离历史区间 |
| `gap` | 开盘跳空,隔夜消息面 |
| `excess_ret` | 相对指数超额收益极端 |

**⚠️ 复权跳变陷阱**（Agent 必须警觉）：

v0.1.0 未做 `|ret| < 0.11` 过滤，若榜单中某只股票 `ret_T` 绝对值 > 0.11（A 股单日理论上限约 20%，>0.11 即异常）：

- 显式提示："<symbol> 的 T 日收益 <值> 疑似复权跳变导致,非真实市场异常,建议忽略"
- 例：20260729 榜单里国电电力 `ret_T=+318`、中国巨石 `+18.9`、兆易创新 `+4.27` 都是复权跳变污染，不是真实异常信号

**数据侧特殊情况**：

- **exit 1（panda_data 5xx）** → "panda_data 服务暂不可用,稍后再试"
- **exit 3（池空）** → "该日无 CSI300 成分股数据,可能是非交易日"
- **训练时间明显偏长（> 3 分钟）** → 可能是 CPU 兜底（MPS 不可用）,不影响结果但告知用户

**收尾一句**（可选）：如果用户看起来还会追问，加"如需看某只股票的 per-feature 分解、换扫描日、或调 Top-N,告诉我"。

## 参数调整（用户主动要求时才调）

用户明确要求换指数、调窗口、放宽 Top-N 时，透传对应 CLI 参数：

```bash
python scripts/scan.py \
    --date 20260729 \
    --index 000300.SH \
    --lookback 20 \
    --train_days 60 \
    --top_n 10 \
    --epochs 50 \
    --batch_size 256 \
    --seed 42
```

否则一律用默认参数。

## 开发者入口（不用于 Agent 触发路径）

```bash
# 字段自检（升级 panda_data 后手动跑一次）
python -m scripts.data --self-check --date 20260729

# 单元测试
pytest tests/ -v
```

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
