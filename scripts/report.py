"""CSV + Markdown emitters for the anomaly leaderboard.

The hits frame produced by scripts/scan.py has the 10 columns declared in SKILL.md
`## 输出结果`; this module orders rows by descending `reconstruction_error` and
writes both formats.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HITS_COLUMNS: list[str] = [
    "trade_date",
    "symbol",
    "name",
    "rank",
    "reconstruction_error",
    "top_feature",
    "ret_T",
    "turnover_T",
    "amplitude_T",
    "detail_json",
]


def _order_and_rank(hits_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Sort by reconstruction_error desc, truncate to top_n, assign 1..N rank."""
    if hits_df.empty:
        return hits_df.assign(rank=[]) if "rank" not in hits_df.columns else hits_df
    df = hits_df.sort_values("reconstruction_error", ascending=False).head(top_n).copy()
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def write_csv(hits_df: pd.DataFrame, path: str, top_n: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = _order_and_rank(hits_df, top_n)
    # Enforce column order + presence.
    for c in HITS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[HITS_COLUMNS]
    df.to_csv(path, index=False)


def _fmt_row_md(row: pd.Series) -> str:
    return (
        f"| {int(row['rank'])} | {row['symbol']} | {row.get('name', '') or ''} | "
        f"{row['reconstruction_error']:.4f} | {row.get('top_feature', '') or ''} | "
        f"{row['ret_T']:+.4f} | {row['turnover_T']:.4f} | {row['amplitude_T']:.4f} |"
    )


def write_markdown(
    hits_df: pd.DataFrame,
    path: str,
    *,
    date: str,
    top_n: int,
    meta: dict,
) -> None:
    """Emit the human-readable markdown report.

    Args:
        meta: dict with training metadata: `index`, `universe_size`, `train_start`,
              `train_end`, `train_samples`, `epochs_ran`, `val_loss`, `device`,
              `error_mean`, `error_std`, `error_max`.
    """
    df = _order_and_rank(hits_df, top_n)

    lines: list[str] = []
    lines.append(f"# AE 异常检测榜单 · {date}\n")
    lines.append(f"- **扫描日**: {date}")
    lines.append(
        f"- **Universe**: {meta.get('index', '000300.SH')} "
        f"（当日 {meta.get('universe_size', '?')} 只，参与训练/打分 {meta.get('score_size', '?')} 只）"
    )
    lines.append(
        f"- **训练窗口**: {meta.get('train_start', '?')} → {meta.get('train_end', '?')} "
        f"（{meta.get('train_days', '?')} 交易日）"
    )
    lines.append(
        f"- **训练样本**: {meta.get('train_samples', '?')} 条"
        f"（剔除停牌/缺失后）"
    )
    lines.append(
        f"- **模型**: MLP AE (160 → 32 → 160)，训练 {meta.get('epochs_ran', '?')} epochs，"
        f"设备 {meta.get('device', 'cpu')}"
    )
    lines.append(
        f"- **验证集 MSE**: {meta.get('val_loss', float('nan')):.4f}"
    )
    lines.append(
        f"- **重建误差分布**: mean={meta.get('error_mean', float('nan')):.4f}, "
        f"std={meta.get('error_std', float('nan')):.4f}, "
        f"max={meta.get('error_max', float('nan')):.4f}\n"
    )

    if df.empty:
        lines.append("\n_今日无有效打分样本（数据可能未就绪或全部停牌）。_\n")
    else:
        lines.append(f"## Top {len(df)} 异常\n")
        lines.append("| Rank | Symbol | Name | Error | 主导特征 | 收益T | 换手T | 振幅T |")
        lines.append("|------|--------|------|-------|----------|-------|-------|-------|")
        for _, r in df.iterrows():
            lines.append(_fmt_row_md(r))

        # Feature-frequency interpretation.
        top_feat_counts = df["top_feature"].value_counts().head(3)
        summary_parts = [f"{feat}（{cnt}/{len(df)}）" for feat, cnt in top_feat_counts.items()]
        lines.append(
            f"\n---\n\n_今日 Top {len(df)} 异常的主导特征分布：{', '.join(summary_parts)}。_\n"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
