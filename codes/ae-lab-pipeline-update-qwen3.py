"""
ae_pipeline_simple_qwen3.py

顺序做三件事，但只输出最终一个结果 CSV：
1) 用本地 Qwen3-8B 从 notes 里抽 AE
2) （可选）用 baseline 过滤掉 baseline 以内的 AE
3) 用你微调好的 MedCPT 模型映射到 CTCAE v5.0

最终输出一个表，列大致为：
MRN, Onset Date, Date Resolved, CTCAE, Grade,
Attr to Disease, AE Immune related?, Serious Y/N,
CTCAE_Mapped_Top1, Similarity_Top1, Final_CTCAE_Term
"""

import os
import time
import json
import re
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from incremental_update import update_patient_history


# ==============================
# 0) Qwen3-8B 本地推理设置
# ==============================
# ✅ 你指定先用这个
QWEN_MODEL_NAME = "Qwen/Qwen3-8B"

# 建议：固定用单卡（GPU0），避免 device_map="auto" 自动切多卡带来不确定性
# 如果你想用别的卡，把 "0" 改成对应编号；或直接注释掉这行让其自动选择
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⏳ Loading Qwen3 model: {QWEN_MODEL_NAME} | device={DEVICE} | dtype={DTYPE}")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).to(DEVICE)
qwen_model.eval()


def qwen_generate_json_array(prompt: str, max_new_tokens: int = 2048) -> str:
    """
    用 Qwen3 生成，尽量保证只返回 JSON array。
    双保险：/no_think + enable_thinking=False，并且后处理清理 <think> 块与多余文本。
    """
    messages = [
        {"role": "system", "content": "You are a clinical research assistant. Return ONLY a JSON array. No explanation."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]

    input_ids = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,  # 关键：关 thinking，避免 <think>...</think>
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # 稳定输出（等价 temperature=0）
        )

    text = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # 清理可能出现的 think 块（保险）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 尝试从文本中截取最后一个 JSON 数组
    # 策略：找最后一个 '[' 和与之匹配的最后一个 ']'
    l = text.rfind("[")
    r = text.rfind("]")
    if l != -1 and r != -1 and r > l:
        text = text[l : r + 1].strip()

    return text


# ================= 1) GPT 提取 AE 的 prompt（保持你原样） =================
base_prompt = """
You are a clinical research assistant helping to extract adverse events (AEs) from clinical notes.
Note:
<text>
{text}
</text>

For each AE, extract the following fields **in JSON array format** (one object per AE):

- MRN (from the note)
- Onset Date: If a specific start date is mentioned, extract it directly; otherwise, use the clinic note date as the start date or estimate an onset date according to the notes. "Onset Date" MUST NEVER be "Unknown" or "unknown". For any AE, if no explicit onset date is mentioned, ALWAYS set "Onset Date" exactly to {doc_date} (or date (estimated)), never to "Unknown".
- Date Resolved: If a specific end date or resolution (“…has resolved”) is mentioned, extract it; for events like “weight loss → gain weight,” use the clinic note date as the end date. If the AE is described as ongoing, set end date to “ongoing.” If not mentioned, set end date to “unknown.”
- AE term (mapped to CTCAE terminology)
- Grade (must be 1 to 5) If grade is not explicitly stated, estimate it based on context (Grade 1 for mild, Grade 2 if moderate/intervention needed, Grade 3 if Severe pain; Grade 4 if Life-threatening).
- Attribution to Disease? One of [Unrelated, Unlikely, Possible, Probable, and Definite]
- Immune-related AE? (Yes/No): Mark “Yes” if the AE is immune-related (irAE) based on the following definition.
Definition of immune-related adverse events (irAEs):irAEs are adverse events relevant to immunotherapy, such as colitis, thyroiditis, hypophysitis, adrenalitis, myositis, myocarditis, encephalitis, pneumonitis, hepatitis, immunotherapy-induced diabetes mellitus, vitiligo, and similar conditions. If the AE is immune-mediated or commonly recognized as an irAE, mark “Yes”; otherwise, mark “No”.
- serious AE? (Yes/No) Mark “Yes” if the AE is considered serious (e.g., life-threatening, hospitalization, or significant disability); otherwise, “No.”

**Important**:

Use the note date (if known) to anchor temporal reasoning.

Do not ignore symptoms that are briefly mentioned, appear together with other events, or are described with mild tone. Even minor or vague symptoms should be extracted.

If multiple symptoms are listed in one sentence, treat them as distinct AEs, and extract each separately.

Also extract imaging-based AEs that are not explicitly labeled as diagnoses but can be inferred from radiology findings such as CT scans.

Note any symptoms or minor symptoms that are mentioned as resolved. it should still be extracted and recorded, with end date set to resolution date if known, or clinic note date otherwise.

If no adverse events are present, return an empty JSON array: []
Do NOT include any explanation. Only return the JSON array.

Return a JSON array. Each AE MUST be a JSON object with EXACTLY the following keys
(using the same spelling and capitalization):

[
  {{
    "MRN": "...",
    "Onset Date": "...",
    "Date Resolved": "...",
    "AE Term": "...",
    "Grade": "...",
    "Attribution to Disease": "...",
    "Immune-related AE": "Yes" or "No",
    "Serious AE": "Yes" or "No"
  }}
]

Use these keys EXACTLY as written.
Do NOT add question marks, extra spaces, or any additional keys.
Do NOT change capitalization.

Patient info:
MRN: {mrn}
Document Date: {doc_date}
Document Name: {doc_name}
"""


# ============= 函数 1：Qwen，从 notes CSV -> AE DataFrame（仅在内存） =============
def llm_extract_ae_qwen(note_csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(note_csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    structured_results = []

    for i, row in df.iterrows():
        prompt = base_prompt.format(
            text=row["Document Text"],
            mrn=row["mrn"],
            doc_date=row["Document Date"],
            doc_name=row["Document Name"],
        )

        print(f"\n=== Qwen Step | row {i} | MRN: {row['mrn']} ===")

        try:
            reply = qwen_generate_json_array(prompt, max_new_tokens=2048)

            # 解析 JSON 数组
            if not (reply.startswith("[") and reply.endswith("]")):
                print(f"⚠️ Row {i}: Not a valid JSON array boundary. Skipping.\nReply head: {reply[:200]}")
                continue

            ae_list = json.loads(reply)
            if not isinstance(ae_list, list) or len(ae_list) == 0:
                print(f"ℹ️ Row {i}: No AE extracted.")
                continue

            for ae in ae_list:
                ae["Document Date"] = row["Document Date"]
                ae["Document Name"] = row["Document Name"]
                structured_results.append(ae)

        except Exception as e:
            print(f"❌ Error on row {i}: {e}")
            continue

        # 本地模型一般不需要 sleep；如果你担心 IO/日志太密可以留
        # time.sleep(0.1)

    ae_df = pd.DataFrame(structured_results)

    if ae_df.empty:
        print("⚠️ LLM 没有抽到任何 AE。")
        return ae_df

    # 标准化几列，后面 filter / mapping 要用
    ae_df["MRN"] = ae_df["MRN"].astype(str).str.strip()
    ae_df["CTCAE"] = ae_df["AE Term"].astype(str).str.strip().str.lower()
    ae_df["Grade"] = pd.to_numeric(ae_df["Grade"], errors="coerce")

    return ae_df


# ============= 函数 2：baseline filter（可选） =============
def filter_with_baseline(ae_df: pd.DataFrame, baseline_file: str | None) -> pd.DataFrame:
    """如果 baseline_file 是 None 或 ""，则直接返回 ae_df，不做任何过滤。"""
    if ae_df.empty:
        return ae_df

    if baseline_file is None or baseline_file == "":
        print("ℹ️ 未提供 baseline 文件，跳过 baseline 过滤。")
        return ae_df

    baseline_df = pd.read_excel(baseline_file)
    baseline_df.columns = baseline_df.columns.str.strip()

    # 关键列名（按你之前的脚本）
    subject_col = "Patient"
    ae_term_col = "Adverse Event Term (v5.0)"
    baseline_grade_col = "Grade"  # baseline AE grade 列
    ae_grade_col = "Grade"        # 我们 AE 表里的 Grade 列

    # baseline 标准化
    baseline_df[subject_col] = baseline_df[subject_col].astype(str).str.strip()
    baseline_df[ae_term_col] = baseline_df[ae_term_col].astype(str).str.strip().str.lower()
    baseline_df[baseline_grade_col] = (
        baseline_df[baseline_grade_col].astype(str).str.extract(r"(\d+)").astype(float)
    )

    merged = ae_df.merge(
        baseline_df[[subject_col, ae_term_col, baseline_grade_col]],
        how="left",
        left_on=["MRN", "CTCAE"],
        right_on=[subject_col, ae_term_col],
        suffixes=("", "_baseline"),
    )

    ae_grade = merged[ae_grade_col]
    baseline_grade = merged["Grade_baseline"].fillna(-1)

    keep_mask = ae_grade > baseline_grade
    filtered_df = merged[keep_mask]

    filtered_df = filtered_df[ae_df.columns]
    print(f"✅ baseline filter：从 {len(ae_df)} 条 AE 保留 {len(filtered_df)} 条")
    return filtered_df


# ============= 函数 3：MedCPT 映射 CTCAE（ae_df -> final_df） =============
def map_to_ctcae_medcpt(
    ae_df: pd.DataFrame,
    ctcae_dict_csv: str,
    medcpt_model_dir: str,
) -> pd.DataFrame:
    if ae_df.empty:
        print("⚠️ 没有 AE 可映射。")
        return ae_df

    # 读取 CTCAE 词表
    ctcae_df = pd.read_csv(ctcae_dict_csv)
    ctcae_df.columns = ctcae_df.columns.str.strip()
    ctcae_terms = (
        ctcae_df["CTCAE Term"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    # 加载 MedCPT
    print("⏳ 加载 MedCPT 模型...")
    tokenizer = AutoTokenizer.from_pretrained(medcpt_model_dir)
    model = AutoModel.from_pretrained(medcpt_model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 编码 CTCAE 词表（保存在 CPU，可以节省显存）
    def encode_list(texts):
        embs = []
        for t in texts:
            t = str(t)
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                norm_emb = F.normalize(cls_emb, p=2, dim=1)
                embs.append(norm_emb[0].cpu())
        return torch.stack(embs)

    print("⏳ 编码 CTCAE 术语...")
    ctcae_embeddings_cpu = encode_list(ctcae_terms)

    # AE → top-3 CTCAE
    print("⏳ 匹配 AE → Top-3 CTCAE ...")
    top_k = 3
    topk_rows = []

    # ⭐ 小优化：把 embeddings 一次性搬到 GPU（H200 显存够，不需要每条搬一次）
    ctcae_embeddings = ctcae_embeddings_cpu.to(device)

    for ctcae_free in ae_df["CTCAE"]:
        inputs = tokenizer(
            ctcae_free,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            ae_emb = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)

        sim = torch.mm(ae_emb, ctcae_embeddings.T).squeeze()
        topk_scores, topk_indices = torch.topk(sim, k=top_k)

        row = {}
        for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores), start=1):
            row[f"CTCAE_Mapped_Top{rank}"] = ctcae_terms[int(idx)].title()
            row[f"Similarity_Top{rank}"] = float(score)
        topk_rows.append(row)

    topk_df = pd.DataFrame(topk_rows)
    df = pd.concat([ae_df.reset_index(drop=True), topk_df.reset_index(drop=True)], axis=1)

    # 精确匹配 + Final_CTCAE_Term
    ctcae_set = set(ctcae_terms)

    def exact_match(term):
        if isinstance(term, str) and term.lower() in ctcae_set:
            return term
        return None

    df["CTCAE_Mapped_Exact"] = df["CTCAE_Mapped_Top1"].apply(exact_match)
    df["CTCAE_Mapped_By"] = df["CTCAE_Mapped_Exact"].apply(lambda x: "exact" if x is not None else "semantic")
    df["Final_CTCAE_Term"] = df["CTCAE_Mapped_Exact"].combine_first(df["CTCAE_Mapped_Top1"])

    # 重命名列
    df = df.rename(
        columns={
            "Attribution to Disease": "Attr to Disease",
            "Immune-related AE": "AE Immune related?",
            "Serious AE": "Serious Y/N",
        }
    )

    # 去重
    df = df.drop_duplicates(
        subset=[
            "MRN", "Onset Date", "CTCAE", "Grade",
            "CTCAE_Mapped_Top1", "CTCAE_Mapped_Top2", "CTCAE_Mapped_Top3",
            "Final_CTCAE_Term"
        ]
    )

    final_cols = [
        "MRN",
        "Onset Date",
        "Date Resolved",
        "CTCAE",
        "Grade",
        "Attr to Disease",
        "AE Immune related?",
        "Serious Y/N",
        "CTCAE_Mapped_Top1", "Similarity_Top1",
        "CTCAE_Mapped_Top2", "Similarity_Top2",
        "CTCAE_Mapped_Top3", "Similarity_Top3",
        "Final_CTCAE_Term",
    ]

    for c in final_cols:
        if c not in df.columns:
            df[c] = ""

    return df[final_cols]


# ============= 4) 整个 pipeline：只输出最终 step3 CSV + history merge =============
def run_pipeline(
    note_csv_path: str,
    baseline_file: str | None,
    ctcae_dict_csv: str,
    medcpt_model_dir: str,
    final_output_csv: str,
    max_rows: int | None = None,   # ✅ 方便你先小跑
):
    # Step 1: Qwen3 抽 AE
    ae_df = llm_extract_ae_qwen(note_csv_path, max_rows=max_rows)

    # Step 2: baseline 过滤（可选）
    ae_filtered = filter_with_baseline(ae_df, baseline_file)

    # Step 3: MedCPT 映射
    final_df = map_to_ctcae_medcpt(ae_filtered, ctcae_dict_csv, medcpt_model_dir)

    # 👉 只有这里写出一个最终文件
    final_df.to_csv(final_output_csv, index=False)
    print(f"\n🎉 全部完成！最终结果 CSV：{final_output_csv}")

    HISTORY_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/ae_history"

    merged_df = update_patient_history(
        ae_new_df=final_df,
        history_dir=HISTORY_DIR,
        mrn_col="MRN",
    )

    # 可选：只用于 debug / 前端临时查看（不是必须）
    latest_path = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/latest_merged.csv"
    merged_df.to_csv(latest_path, index=False)
    print(f"✅ latest merged saved -> {latest_path}")

    return merged_df


# ============= 5) 直接跑脚本用 =============
if __name__ == "__main__":

    NOTES_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/reversed_Clindoc266614-1_05_progress_note.csv"
    BASELINE_XLSX = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/18C0056_BL_Subgroup_02.xlsx"
    CTCAE_DICT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/CTCAE_v5.0.csv"
    MEDCPT_MODEL_DIR = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/medcpt_ctcae_triplet_epoch10"

    FINAL_OUTPUT_CSV = "/netmnt/vast01/cbb01/lulab/gey2/AE_extraction/AE_Extraction_CTCAE_Map/qwen3_8b_top3_pipeline_1217_from_extracted_ae.csv"

    # ✅ 先小跑几行看看 JSON 稳不稳；稳定后把 max_rows=None
    run_pipeline(
        note_csv_path=NOTES_CSV,
        baseline_file=BASELINE_XLSX,
        ctcae_dict_csv=CTCAE_DICT_CSV,
        medcpt_model_dir=MEDCPT_MODEL_DIR,
        final_output_csv=FINAL_OUTPUT_CSV,
        max_rows=1,
    )
