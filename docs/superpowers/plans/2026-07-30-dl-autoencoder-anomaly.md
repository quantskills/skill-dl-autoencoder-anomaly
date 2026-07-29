# skill-dl-autoencoder-anomaly — Implementation Plan

## Context

在 `/Users/since/Code/quantskills/` 下新建第三个 skill:**基于深度自编码器 (MLP AE) 的沪深 300 无监督异常检测**。

- 现有两个 skill (`skill-etf-flow-radar`, `skill-portfolio-blacklitterman`) 均已成熟落地,新 skill 需**对齐其目录结构、CLI 风格、references 组织和验收测试范式**,以维持工程一致性。
- 与前两个 skill 的关键差异:本 skill 引入 **PyTorch 训练步骤** —— 首个含 ML 训练环节的 skill。
- 期望产出:每晚一条命令,输入交易日 `T` (默认最近交易日),对沪深 300 每只成分股基于近 60 日多维特征进行 AE 重建,输出重建误差 Top-10 的异常股票榜单 (CSV + Markdown)。用户后续可基于榜单深度调研。

用户明确的关键决策:
1. **异常范围**: 横截面 + 时序混合,universe = CSI 300 (`000300.SH`)
2. **模型**: 深 MLP Autoencoder (160 → 96 → 48 → 32 → 48 → 96 → 160)
3. **训练策略**: 每次运行都用 T-60 到 T-1 的短窗口重训 (不落盘 checkpoint,每次运行 ≈ 30 秒 CPU)
4. **输出**: Top-10 异常股 (按重建误差 desc),CSV + Markdown 报告
5. **验证**: 只输出重建误差,不合并龙虎榜 / 涨跌停等外部事件 (留 v0.2)
6. **运行环境**: conda env `pandaai` + `pip install torch`

---

## 目录结构 (镜像 skill-etf-flow-radar,补 models/ 与 train 模块)

```
skill-dl-autoencoder-anomaly/
├── SKILL.md                     # 主契约 (中文,YAML frontmatter + 章节)
├── README.md                    # 极简英文入口 (env, install, run)
├── skill.json                   # {name, description, version:"0.1.0", tags, scripts}
├── requirements.txt             # pandas, panda_data, torch, pytest
├── .gitignore                   # 忽略 output/, models/, __pycache__
├── references/
│   └── need_used_api.md         # 从 panda_data_api_doc.md 抽出的所需 API 契约
├── scripts/
│   ├── __init__.py
│   ├── data.py                  # panda_data 封装 + init + self_check + 列校验
│   ├── universe.py              # CSI300 成分股取数 (get_index_weights on T)
│   ├── features.py              # 特征工程:8 维 × 20 日 → 160 维扁平向量
│   ├── model.py                 # torch.nn.Module MLP AE 定义
│   ├── train.py                 # 训练循环 (Adam + early stopping)
│   ├── scan.py                  # 主 CLI: fetch → build features → train → score → report
│   └── report.py                # write_csv, write_markdown
├── tests/
│   ├── conftest.py              # sys.path 挂载
│   ├── test_data.py             # panda_data 封装的 mocking 测试
│   ├── test_universe.py         # 成分股取数 + 边界情况
│   ├── test_features.py         # 特征工程数学正确性 + no-lookahead
│   ├── test_model.py            # AE 前向 shape + 参数量
│   └── test_train.py            # 迷你数据集上的收敛烟雾测试
├── output/                      # 运行时产物:anomaly_YYYYMMDD.{csv,md}
└── docs/superpowers/
    ├── specs/2026-07-30-dl-autoencoder-anomaly-design.md
    └── plans/2026-07-30-dl-autoencoder-anomaly.md
```

---

## 数据管道 (从 panda_data 抽取)

### 步骤 A:确定扫描日 T 和训练日范围

- `get_last_trade_date(exchange="SH")` → 最新交易日 (若 CLI `--date` 未指定)
- `get_prev_trade_date(date=T, n=60)` → 训练窗口起点

### 步骤 B:锁定 CSI 300 成分股 (universe)

- **首选**: `get_index_weights(index_symbol="000300.SH", start_date=T, end_date=T)` → 300 只当日成分。
- **回退**: 若 `get_index_weights` 无 T 日数据 (可能因为权重发布滞后),取最近发布日的成分。

### 步骤 C:批量取 61 个交易日的后复权日线 (T-60 到 T)

- `get_stock_daily_post(start_date=T-60, end_date=T, symbol=constituents, st=False)`
  - 返回列: `date, symbol, name, open, close, high, low, volume, pre_close, limit_up, limit_down, trade_status`
- **同步**取 `get_stock_daily(...)` 拿 `amount` (post 版无 amount) —— 或直接用 `get_factor(..., factors=["turnover","market_cap"])` 一次性拿 `close/volume/amount/turnover/market_cap`。**决定**:统一走 `get_factor`,省一次 join。
- 取市场基准: `get_index_daily(symbol="000300.SH", start_date=T-60, end_date=T)` 计算超额收益。

### 步骤 D:特征工程 (features.py)

对每只股票、每个交易日,构造 **8 维特征**:

| # | 名称 | 公式 | 说明 |
|---|------|------|------|
| 1 | `ret` | `close / pre_close - 1` | 日收益 |
| 2 | `log_vol` | `log(volume + 1)` | 对数成交量 |
| 3 | `amplitude` | `(high - low) / pre_close` | 振幅 |
| 4 | `turnover` | `turnover` (from get_factor) | 换手率 |
| 5 | `gap` | `open / pre_close - 1` | 开盘跳空 |
| 6 | `dist_limit_up` | `(limit_up - close) / close` | 距涨停 |
| 7 | `dist_limit_down` | `(close - limit_down) / close` | 距跌停 |
| 8 | `excess_ret` | `ret - index_ret` | 相对 CSI300 超额收益 |

- **样本构造**: 对每只股票 × 每个训练日 `t ∈ [T-60, T-1]`,取 `t-19` 到 `t` 的 20 日窗口 → 20×8=160 维向量。总样本 ≈ 300 × 40 ≈ 12,000 (少于 T-60 到 T-1 是因为要留 20 日历史)。
- **T 日打分样本**: 每只股票取 `T-19` 到 `T` 的 20 日窗口 → 300 条打分向量。
- **标准化**: 训练集上按特征列 (跨股票、跨时间) 计算 `mean/std`,z-score 标准化;同样的 mean/std 应用于 T 日打分样本 (防泄露)。
- **缺失处理**: 停牌日 (`trade_status != 0`) 或数据缺失的股票,若训练窗口内缺失日 > 5,整只剔除;若打分窗口缺失,不出现在榜单。

### 步骤 E:no-lookahead 保证

- 训练集**严格不包含 T 日**特征。用字符串比较 `df[df["date"] < date_T]` 显式过滤 (对齐 etf-radar 的 s7 idiom)。
- 有对应测试 `test_features.py::test_no_lookahead`。

---

## 模型 (model.py + train.py)

### 模型结构 (model.py)

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=160, hidden=[96, 48], code_dim=32, dropout=0.1):
        # encoder: 160 → 96 → 48 → 32
        # decoder: 32 → 48 → 96 → 160
        # 每层 Linear + ReLU + Dropout,decoder 最后一层不加 ReLU
```

- 输入维度 160,压缩到 32 维 codes,5x 压缩比。
- 参数量约 (160×96 + 96×48 + 48×32) × 2 + biases ≈ 45k,轻量足以在 CPU 上快速收敛。

### 训练循环 (train.py)

- **损失**: MSE (逐元素均方误差)
- **优化器**: Adam,lr=1e-3
- **Batch**: 256
- **Epochs**: max 50,early stopping patience=5 (验证集重建 MSE 停止下降 5 轮就停)
- **划分**: 训练集 12k 样本随机 80/20 训练/验证
- **随机种子**: `torch.manual_seed(42)` + `numpy.random.seed(42)`,保证复现
- **设备**: 自动检测 `cuda` / `mps` / `cpu`,默认 CPU 上跑

### 打分 (scan.py)

- 对 T 日的 300 条打分向量,前向 → 重建 → 计算 per-sample MSE
- 排序 desc,取 Top-10
- 输出:`symbol, name, reconstruction_error, ret_T, turnover_T, amplitude_T, top_feature`
  - `top_feature`: 分解 per-feature-column MSE,取贡献最大的特征名 (帮助解释异常)

---

## CLI 契约 (scripts/scan.py)

```bash
# 基础用法 (默认 CSI300, 最近交易日)
python scripts/scan.py

# 指定日期
python scripts/scan.py --date 20260729

# 指定 universe (未来可切 CSI500)
python scripts/scan.py --index 000905.SH

# 完整参数
python scripts/scan.py \
    --date 20260729 \
    --index 000300.SH \
    --lookback 20 \
    --train_days 60 \
    --top_n 10 \
    --epochs 50 \
    --seed 42

# 字段自检 (对齐 etf-radar 的 self-check 模式)
python -m scripts.data --self-check --date 20260729

# 单测
pytest -q
```

**退出码约定 (对齐 etf-radar):**
- 0: 成功
- 1: panda_data 错误 / 网络
- 2: 指定日期无数据
- 3: CSI300 成分股为空
- 4: 列自检失败

---

## 输出契约

### CSV: `output/anomaly_YYYYMMDD.csv`

列 (10 列):
```
trade_date, symbol, name, rank, reconstruction_error, top_feature,
ret_T, turnover_T, amplitude_T, detail_json
```
- `detail_json`: `{"per_feature_mse": {...}, "z_score": <float>, "trained_samples": <int>}`
- 排序:`reconstruction_error` desc

### Markdown: `output/anomaly_YYYYMMDD.md`

```markdown
# AE 异常检测榜单 — 20260729

- 扫描日: 20260729
- Universe: 沪深300 (000300.SH,当日 300 只)
- 训练窗口: 20260430 → 20260728 (60 交易日)
- 训练样本: 11,847 条 (剔除停牌/缺失后)
- 模型: MLP AE (160 → 32 → 160),训练 32 epochs
- 重建误差 z-score 分布: mean=0, std=1, max=<X>

## Top 10 异常

| Rank | Symbol | Name | Error | 主导特征 | 收益T | 换手T | 振幅T |
|------|--------|------|-------|----------|-------|-------|-------|
| 1    | ...    | ...  | ...   | ...      | ...   | ...   | ...   |

---
解读: <一句话>: 今日异常显著,以 <top_feature> 类异常为主 (N/10 只)。
```

---

## references/need_used_api.md 内容

从 `/Users/since/Code/quantskills/panda_data_api_doc.md` 抽出 **6 个 API 的原文契约段** (格式对齐 etf-flow-radar 的 `**N. func_name**` 编号 + 参数表 + 响应表 + Python 例子 + 样例返回):

1. `get_last_trade_date` — 拿最新交易日
2. `get_prev_trade_date` — 计算 T-60 起点
3. `get_index_weights` — CSI300 成分股锁定
4. `get_factor` — OHLCV + turnover + market_cap 一次搞定
5. `get_stock_daily_post` — 后复权日线 (备用,若 get_factor 无法拿 pre_close/limit_up/down)
6. `get_index_daily` — 基准指数收益

**决策**:主打 `get_factor`,只有当 `get_factor` 不返回 `pre_close/limit_up/limit_down/trade_status` 时才回退到 `get_stock_daily_post`。首次实现时先 stub 一份可能的字段,跑一次 self-check 探明 `get_factor` 的实际返回列,再决定是否需要 join `get_stock_daily_post`。

---

## 测试要点

### test_features.py (关键)
- `test_feature_shape`: 输入 20 日 8 特征 → 160 维向量
- `test_no_lookahead`: 训练矩阵严格 `date < T`
- `test_amplitude_formula`: 振幅公式正确
- `test_dist_limit_signs`: 距涨停恒 ≥ 0,距跌停恒 ≥ 0
- `test_excess_ret_zero_when_stock_matches_index`: 用假指数验证

### test_model.py
- `test_ae_forward_shape`: 前向输出维度 = 输入维度
- `test_param_count`: 参数量在合理范围 (≈45k)
- `test_encoder_bottleneck`: 中间层输出维度 = 32

### test_train.py
- `test_train_smoke`: 用一个 100 样本 × 160 维的假数据训练 3 epochs,损失单调下降
- `test_seed_reproducibility`: 相同 seed 两次训练,最终损失差 < 1e-6

### test_data.py + test_universe.py
- panda_data 用 `sys.modules` mock,验证 (a) auth 缺 env var 报错,(b) 列自检抓字段缺失,(c) get_index_weights 空返回 → 抛 Exit 3

---

## 关键复用点 (来自 skill-etf-flow-radar)

对照 Explore agent 报告,以下 idiom 直接搬:

| ETF-radar 位置 | 复用到 anomaly skill |
|----------------|----------------------|
| `scripts/data.py::init_panda_data()` — env-var auth | `scripts/data.py` 里同样 lazy import |
| `scripts/data.py::EXPECTED_COLUMNS` + `_assert_columns` | 复制过来,换成 anomaly 用到的字段 |
| `scripts/data.py::_main()` 的 `--self-check` argparse | 完全一样的模式 |
| `scripts/universe.py` — pure function 返回 (list, enrichment_df) | 我们的返回 (constituents_list, weights_df) |
| `scripts/radar.py::_resolve_scan_date` | 直接搬,只是数据源换成 daily 而不是 flow |
| `scripts/radar.py` 顶部 `sys.path.insert(0, ...)` 补丁 | 一模一样 |
| `scripts/report.py` — `SIGNAL_ORDER` + markdown 拼接 idiom | 我们的 report 更简单 (只有一个 Top-10 表) |
| `tests/conftest.py` 6 行 sys.path 挂载 | 复制不改 |
| exit-code convention (0/1/2/3/4) | 完全对齐 |

**不能复用的部分** (Autoencoder 独有):
- `scripts/model.py`, `scripts/train.py`, `scripts/features.py` 完全新写
- `requirements.txt` 增加 `torch`
- `SKILL.md` 加运行环境章节:必须 `conda activate pandaai` + `pip install torch`

---

## 交付节奏 (12 个 task,略多于 etf-radar 的 11 个,因为多了 model + train + features 三块)

1. 生成 skill 骨架 (SKILL.md 主框架 + README + skill.json + requirements.txt + .gitignore + 目录)
2. 抽取 references/need_used_api.md (从 panda_data_api_doc 挑出 6 个 API 段落原文)
3. 实现 scripts/data.py + test_data.py + `--self-check` CLI
4. 实现 scripts/universe.py + test_universe.py (CSI300 取数)
5. 实现 scripts/features.py + test_features.py (8 特征 × 20 日,含 no-lookahead 测试)
6. 实现 scripts/model.py + test_model.py (MLP AE 网络结构)
7. 实现 scripts/train.py + test_train.py (Adam + early stopping)
8. 实现 scripts/scan.py 主 CLI (装载→特征→训练→打分→写盘)
9. 实现 scripts/report.py (CSV + Markdown 写盘)
10. 集成测试:用真实 panda_data 跑一次 T=20260729,肉眼审 output/
11. 补文档:`SKILL.md` 三章补齐 (数据接口 / 术语约定 / 验收要求 / 已知局限 v0.1.0)
12. Git 初始化 + tag v0.1.0 + 写 memory 文件

---

## 验收标准 (对齐 etf-radar `## 验收要求`)

1. 无未来函数: 所有 `date < T` 显式过滤,`test_no_lookahead` 通过
2. `pytest -q` 全绿,总数 ≥ 20 (etf-radar 22 条,我们至少同量级)
3. `python -m scripts.data --self-check --date <recent>` 无字段缺失
4. `python scripts/scan.py --date <recent>` 端到端跑通,产出 `output/anomaly_<date>.csv` 与 `.md`,Top-10 完整,无 NaN
5. SKILL.md 引用的每个字段都在 references/need_used_api.md 出现
6. 训练可复现: 相同 seed 两次运行,Top-10 顺序一致

---

## 已知局限 (v0.1.0)

- Universe 固定 CSI300 (300 只);v0.2 支持 CSI500/CSI1000 + 按行业训练
- 每次运行重训,不做 checkpoint;v0.2 支持模型缓存
- 只输出重建误差,不融合龙虎榜 / 涨跌停信号;v0.2 增加"官方异常"对比表
- 单模型全市场;v0.2 支持按 L1 行业训练 31 个子模型
- Dropout 唯一正则化手段;v0.2 增加 β-VAE 变体便于挑选异常方向

---

## 端到端验证方案 (verification)

在 conda pandaai env 下:

```bash
cd /Users/since/Code/quantskills/skill-dl-autoencoder-anomaly
pip install -r requirements.txt

# 1. 字段自检
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
python -m scripts.data --self-check --date 20260729   # 期望 exit 0

# 2. 单测
pytest -q                                              # 期望 ≥20 passed, 0 failed

# 3. 端到端
python scripts/scan.py --date 20260729                 # 期望 exit 0, 产出 CSV + MD

# 4. 复现性
python scripts/scan.py --date 20260729 --seed 42
python scripts/scan.py --date 20260729 --seed 42       # 两次 Top-10 顺序完全一致

# 5. 肉眼审
cat output/anomaly_20260729.md                         # Top-10 表 + 一句话解读
```

同时验证 Explore agent 报告里那条 memory:自 2026-06-11 后 panda_data ETF 折溢价停供 —— 本 skill 不依赖 ETF 折溢价数据,只需要普通 A 股 daily / factor / index,应不受影响。首次运行时若某 API 有异样,记入 memory。
