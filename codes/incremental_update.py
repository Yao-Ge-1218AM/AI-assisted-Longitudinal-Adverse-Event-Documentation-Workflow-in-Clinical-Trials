import os
import pandas as pd
from pathlib import Path

# =========================================================
# 0) 工具：日期解析（只改这里：兼容 5/28/19、2018、10/2018 等）
# =========================================================
def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    强鲁棒日期解析：
    - 优先吃两位年：5/28/19
    - 再吃四位年：05/28/2019
    - 再兜底通用解析
    - 支持：2018, 10/2018
    - ongoing/unknown/空 -> NaT
    """
    s = series.astype(str).str.strip()

    s = s.replace({
        "": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA,
        "unknown": pd.NA, "Unknown": pd.NA,
        "ongoing": pd.NA, "Ongoing": pd.NA
    })

    # ✅ 关键：先强制按两位年解析（专门解决 5/28/19 变 NaT）
    dt_yy = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    dt_yyyy = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    dt_any = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    dt = dt_yy.combine_first(dt_yyyy).combine_first(dt_any)

    # 兜底：2018 -> 2018-01-01
    mask_year = s.notna() & s.str.fullmatch(r"\d{4}") & dt.isna()
    if mask_year.any():
        dt.loc[mask_year] = pd.to_datetime(s[mask_year] + "-01-01", errors="coerce")

    # 兜底：10/2018 -> 2018-10-01
    mask_month_year = s.notna() & s.str.fullmatch(r"\d{1,2}/\d{4}") & dt.isna()
    if mask_month_year.any():
        dt.loc[mask_month_year] = pd.to_datetime("01/" + s[mask_month_year], errors="coerce")

    return dt


def _normalize_str(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

# =========================================================
# 1) 你的全局 merge 逻辑（改造成函数）
#    - 输入：一个 DataFrame（包含历史 + 新增）
#    - 输出：合并后的 DataFrame（按 MRN + CTCAE + Grade）
# =========================================================
def merge_ae_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    按你之前的规则：
    - 分组：MRN + CTCAE + Grade
    - Onset Date：最早
    - Date Resolved：具体日期优先；否则 ongoing；否则 unknown
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # 关键列确保存在
    must_cols = ["MRN", "CTCAE", "Grade", "Onset Date", "Date Resolved"]
    for c in must_cols:
        if c not in df.columns:
            df[c] = ""

    # 标准化
    df["MRN"] = df["MRN"].astype(str).str.strip()
    df["CTCAE"] = df["CTCAE"].astype(str).str.strip().str.lower()
    df["Grade"] = df["Grade"].astype(str).str.strip()

    df["Onset Date"] = _safe_to_datetime(df["Onset Date"])

    # Date Resolved 既要保留文本，也要尝试 parse
    df["Date Resolved Text"] = df["Date Resolved"].astype(str).str.strip().str.lower()
    df["Resolved Parsed"] = _safe_to_datetime(df["Date Resolved"])

    # 分组 merge
    merged_rows = []
    grouped = df.groupby(["MRN", "CTCAE", "Grade"], sort=False)

    print(df.loc[df["CTCAE"].isin(["difficulty concentrating","decreased interest in activities"]), ["CTCAE","Onset Date"]].head(20).to_string(index=False))

    for _, group in grouped:
        #row = group.iloc[0].copy()
	# 优先选有 Onset Date 的行作为 base
	base = group[group["Onset Date"].notna()]
	if not base.empty:
    		row = base.iloc[0].copy()
	else:
    		row = group.iloc[0].copy()

        # Onset Date: 最早
        onset_dates = group["Onset Date"].dropna()
        row["Onset Date"] = onset_dates.min() if not onset_dates.empty else pd.NaT

        # Date Resolved: 具体日期 > ongoing > unknown
        resolved_dates = group["Resolved Parsed"].dropna()
        if not resolved_dates.empty:
            row["Date Resolved"] = resolved_dates.max().strftime("%m/%d/%Y")
        elif (group["Date Resolved Text"] == "ongoing").any():
            row["Date Resolved"] = "ongoing"
        else:
            row["Date Resolved"] = "unknown"

        merged_rows.append(row)

    out = pd.DataFrame(merged_rows)

    # 清理辅助列
    for c in ["Date Resolved Text", "Resolved Parsed"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    # 你希望输出的列（如果缺就补空）
    keep_cols = [
        "MRN", "Onset Date", "Date Resolved", "CTCAE", "Grade",
        # 下面这些列：notes pipeline 有、lab pipeline 没有也没关系，会补空
        "Attr to Bintrafusp alfa", "Attr to Disease", "Attr to Other",
        "AE Immune related?", "Serious Y/N",
        # mapping 输出（如果你希望历史里也保存）
        "CTCAE_Mapped_Top1", "Similarity_Top1", "Final_CTCAE_Term"
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = ""

    out = out[keep_cols]

    # Onset Date 转回字符串（给前端更友好）
    out["Onset Date"] = pd.to_datetime(out["Onset Date"], errors="coerce")
    out["Onset Date"] = out["Onset Date"].dt.strftime("%m/%d/%Y").fillna("")

    # 去重兜底（避免重复写入）
    out = out.drop_duplicates(subset=["MRN", "CTCAE", "Grade", "Onset Date", "Date Resolved"], keep="first")

    return out


# =========================================================
# 2) 增量更新：每个病人一个 history 文件
# =========================================================
def update_patient_history(
    ae_new_df: pd.DataFrame,
    history_dir: str,
    mrn_col: str = "MRN",
) -> pd.DataFrame:
    """
    输入：ae_new_df（本次“当天一条note / 一个lab panel”的最终结果）
    作用：
      - 按 MRN 拆分
      - 对每个 MRN：读取历史文件（若存在）+ 拼接本次结果 + merge_ae_records
      - 保存回 history_dir/{MRN}.csv
    输出：合并后的（所有 MRN）DataFrame（便于前端展示）
    """

    if ae_new_df is None or ae_new_df.empty:
        print("ℹ️ ae_new_df 为空：不更新 history。")
        return pd.DataFrame()

    ae_new_df = ae_new_df.copy()
    ae_new_df.columns = ae_new_df.columns.str.strip()

    # 允许你用 hash 当 MRN，只要在传入前把列名改成 MRN 即可
    if mrn_col not in ae_new_df.columns:
        raise ValueError(f"ae_new_df 缺少列 `{mrn_col}`，无法按病人更新。")

    Path(history_dir).mkdir(parents=True, exist_ok=True)

    all_merged = []

    for mrn, sub in ae_new_df.groupby(mrn_col):
        mrn = str(mrn).strip()
        if mrn == "" or mrn.lower() == "nan":
            print("⚠️ 跳过 MRN 为空的一组")
            continue

        history_path = os.path.join(history_dir, f"{mrn}.csv")

        if os.path.exists(history_path):
            old = pd.read_csv(history_path)
            old.columns = old.columns.str.strip()
            combined = pd.concat([old, sub], ignore_index=True)
        else:
            combined = sub

        merged = merge_ae_records(combined)

        merged.to_csv(history_path, index=False)
        print(f"✅ Updated AE history: {history_path}  (rows={len(merged)})")

        all_merged.append(merged)

    if not all_merged:
        return pd.DataFrame()

    return pd.concat(all_merged, ignore_index=True)


# =========================================================
# 3) 用法示例（你在自己的 notes/lab pipeline 最后调用它）
# =========================================================
if __name__ == "__main__":
    # 假设这是你 pipeline 最终输出给前端看的 final_df（单条 note 的结果）
    # final_df 必须包含至少：MRN, CTCAE, Grade, Onset Date, Date Resolved
    # 其他列没有也没关系，会自动补空
    final_df = pd.read_csv("pipeline_ae_with_ctcae_note2.csv")

    merged_df = update_patient_history(
        ae_new_df=final_df,
        history_dir="./ae_history",   # 每个病人一个文件
        mrn_col="MRN",
    )

    # 这里 merged_df 就是“更新后的最新 AE list”，可以直接 return 给前端
    print("\n=== Latest merged AE list (preview) ===")
    print(merged_df.head(30).to_string(index=False))
