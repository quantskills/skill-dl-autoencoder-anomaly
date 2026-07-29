# panda_data — Autoencoder Anomaly Skill 使用的 API

以下 6 个 API 是本 skill 所依赖的全部数据接口。字段名与参数格式与 `panda_data_api_doc.md` 原文一致。

> **全局约定**
> - 日期格式统一 `YYYYMMDD` 字符串
> - 股票代码带交易所后缀：`.SH` / `.SZ`
> - `panda_data` 为私有包，需 `init_token(username, password)` 后使用
> - 未特别说明的响应表已省略与本 skill 无关的字段

---

**1. get_last_trade_date - 获取最新交易日**

**1.1. 方法名：get_last_trade_date**

**1.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| exchange | Optional[string] | 交易所代码，默认为 "SH"，目前支持"SH"，"HK"和"US" | 非必填 |

**1.3. 响应参数**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 最新交易日，格式 "YYYYMMDD"，如果没有则返回None |

**1.4. 使用示例**

```python
import panda_data
result = panda_data.get_last_trade_date(exchange="SH")
print(result)
```

**响应示例**

```text
date
0  20251223
```

---

**2. get_prev_trade_date - 获取指定日期的前第n个交易日**

**2.1. 方法名：get_prev_trade_date**

**2.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| date | string | 基准日期，格式为 "YYYYMMDD" | 必填 |
| exchange | Optional[string] | 交易所代码，默认为 "SH"，目前支持"SH"，"HK"和"US" | 非必填 |
| n | Optional[integer] | 前第n个交易日，默认为1 | 非必填 |

**2.3. 响应参数**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 第前n个交易日，格式 "YYYYMMDD"，如果没有则返回None |

**2.4. 使用示例**

```python
import panda_data
result = panda_data.get_prev_trade_date(date="20250102", exchange="SH", n=5)
print(result)
```

**响应示例**

```text
date
0  20241225
```

---

**3. get_index_weights - 获取指数权重信息数据**

**3.1. 方法名：get_index_weights**

**3.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| index_symbol | Optional[Union[string, List[string]]] | 指数代码 | 非必填 |
| stock_symbol | Optional[Union[string, List[string]]] | 成分股代码 | 非必填 |
| start_date | string | 开始日期,eg:"20250702" | 必填 |
| end_date | string | 结束日期,eg:"20250702" | 必填 |
| fields | Optional[Union[string, List[string]]] | 返回字段列表 | 非必填 |

**3.3. 响应参数**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| index_symbol | string | 指数代码 |
| date | string | 日期 |
| stock_symbol | string | 股票代码 |
| weight | float | 权重 |

**3.4. 使用示例**

```python
import panda_data
result = panda_data.get_index_weights(
    index_symbol="000300.SH",
    start_date="20260729",
    end_date="20260729",
)
print(result)
```

**响应示例**

```text
index_symbol  date  stock_symbol  weight
0  000300.SH  20260729  600519.SH  0.05
1  000300.SH  20260729  601318.SH  0.04
...
```

---

**4. get_factor - 获取回测因子（股票 OHLCV + turnover + market_cap 一次到位）**

**4.1. 方法名：get_factor**

**4.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| start_date | string | 开始日期,eg:"20250702" | 必填 |
| end_date | string | 结束日期,eg:"20250702" | 必填 |
| symbol | Optional[Union[string, List[string]]] | 股票代码 | 非必填 |
| factors | Union[string, List[string]] | 因子列表 | 必填 |
| type | Optional[string] | 产品类型，支持"stock","future"，默认为"stock" | 非必填 |
| index_component | Optional[string] | 股票池 (e.g. "000300")，默认为空表示查询所有 | 非必填 |

**4.3. 响应参数（股票）**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 日期 |
| symbol | string | 标的代码 |
| name | string | 股票名称 |
| open | float | 开盘价 |
| close | float | 收盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| volume | float | 成交量 |
| amount | float | 成交额 |
| market_cap | float | 市值 |
| turnover | float | 换手率 |

**4.4. 使用示例**

```python
import panda_data
result = panda_data.get_factor(
    start_date="20260601",
    end_date="20260729",
    factors=["open", "close", "high", "low", "volume", "amount", "turnover", "market_cap"],
    type="stock",
    index_component="000300",
)
print(result.head())
```

**响应示例**

```text
date  symbol  open  close  high  low  volume  amount  turnover  market_cap
0  20260729  000001.SZ  ...
```

---

**5. get_stock_daily_post - 获取A股后复权日线数据（补 get_factor 缺列）**

**5.1. 方法名：get_stock_daily_post**

**5.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| start_date | string | 开始日期,eg:"20250702"，与结束日期间不超过5年 | 必填 |
| end_date | string | 结束日期,eg:"20250702"，与开始日期间不超过5年 | 必填 |
| symbol | Optional[Union[string, List[string]]] | 股票 | 非必填 |
| fields | Optional[Union[string, List[string]]] | 返回字段 | 非必填 |
| indicator | Optional[string] | 股票池 (e.g. "000300")，默认为空 | 非必填 |
| st | Optional[bool] | 是否包含ST股，默认True | 非必填 |

**5.3. 响应参数**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 日期 |
| symbol | string | 股票代码 |
| name | string | 股票名称 |
| open | float | 当日开盘价 |
| close | float | 当日收盘价 |
| high | float | 当日最高价 |
| low | float | 当日最低价 |
| volume | float | 当日成交量 |
| pre_close | float | 昨收价 |
| limit_up | float | 当日涨停价 |
| limit_down | float | 当日跌停价 |
| trade_status | integer | 当日是否停牌（0表示当日不停牌） |

**5.4. 使用示例**

```python
import panda_data
result = panda_data.get_stock_daily_post(
    symbol=["000001.SZ"],
    start_date="20260601",
    end_date="20260729",
    indicator="000300",
    st=False,
)
print(result.head())
```

**响应示例**

```text
symbol  date  open  high  low  close  volume  pre_close  limit_up  limit_down  name  trade_status
0  000001.SZ  20260729  ...
```

**说明**：本 skill 从 `get_stock_daily_post` 拿 `pre_close, limit_up, limit_down, trade_status` 四列（`get_factor` 不返回），OHLCV 主用 `get_factor`。两个数据源都是**后复权**，直接 join 不需要再对齐。

---

**6. get_index_daily - 获取指数日线（基准超额收益）**

**6.1. 方法名：get_index_daily**

**6.2. 入参**

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| start_date | string | 开始日期,eg:"20250702"，与结束日期间不超过5年 | 必填 |
| end_date | string | 结束日期,eg:"20250702"，与开始日期间不超过5年 | 必填 |
| symbol | string | 指数代码 | 非必填 |
| fields | string | 返回字段 | 非必填 |

**6.3. 响应参数**

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| symbol | string | 指数代码 |
| date | string | 日期 |
| open | float | 开盘价 |
| close | float | 收盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| volume | float | 成交量 |
| pre_close | float | 昨日结算价 |
| amount | float | 成交额 |

**6.4. 使用示例**

```python
import panda_data
result = panda_data.get_index_daily(
    symbol="000300.SH",
    start_date="20260601",
    end_date="20260729",
)
print(result.head())
```

**响应示例**

```text
symbol  date  amount  close  high  low  open  pre_close  volume
0  000300.SH  20260729  ...
```

**说明**：`excess_ret = stock_ret − index_ret`，其中 `index_ret = index_close / index_pre_close - 1`。用 `--index` 参数指定的指数代码。
