import pandas as pd
from ae_note_pipeline_update_top3v import (
    filter_with_baseline,
    map_to_ctcae_medcpt,
)
from incremental_update import update_patient_history

# ======================
# 1. 读你已经跑完的 AE extraction CSV
# ======================
EXTRACTED_AE_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/Dec_results/gpt4o_1210_Clindoc266614-1_05_PN_structured_ae.csv"
ae_df = pd.read_csv(EXTRACTED_AE_CSV)
ae_df.columns = ae_df.columns.str.strip()

# ---- 统一列名（如果需要）----
rename_map = {
    "AE Term": "CTCAE",
    "Immune-related AE": "AE Immune related?",
    "Serious AE": "Serious Y/N",
}
ae_df = ae_df.rename(columns={k: v for k, v in rename_map.items() if k in ae_df.columns})

# 标准化（和你 GPT step 后做的一样）
ae_df["MRN"] = ae_df["MRN"].astype(str).str.strip()
ae_df["CTCAE"] = ae_df["CTCAE"].astype(str).str.strip().str.lower()
ae_df["Grade"] = pd.to_numeric(ae_df["Grade"], errors="coerce")

print(f"Loaded extracted AE rows = {len(ae_df)}")

# ======================
# 2. baseline filter（可选）
# ======================
BASELINE_XLSX = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/18C0056_BL_Subgroup_02.xlsx"

ae_filtered = filter_with_baseline(
    ae_df=ae_df,
    baseline_file=BASELINE_XLSX,   # 没有就写 None
)

# ======================
# 3. MedCPT 映射
# ======================
CTCAE_DICT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/CTCAE_v5.0.csv"
MEDCPT_MODEL_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/medcpt_ctcae_triplet_epoch10"

final_df = map_to_ctcae_medcpt(
    ae_df=ae_filtered,
    ctcae_dict_csv=CTCAE_DICT_CSV,
    medcpt_model_dir=MEDCPT_MODEL_DIR,
)

# ======================
# 4. 更新 patient-level history（增量）
# ======================
HISTORY_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/ae_history_test"

merged_df = update_patient_history(
    ae_new_df=final_df,
    history_dir=HISTORY_DIR,
    mrn_col="MRN",
)

# ======================
# 5. 输出
# ======================
OUT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/pipeline_1217_from_extracted_ae.csv"
merged_df.to_csv(OUT_CSV, index=False)

print(f"🎉 Done! Final merged AE list saved to:\n{OUT_CSV}")
