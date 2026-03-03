
"""
ae_extraction_pipeline.py

A thin wrapper to run either the NOTES pipeline or the LABS pipeline with a stable interface
for your frontend/backend colleague.

This wrapper avoids hard-coded HISTORY_DIR and "latest_merged.csv" paths inside the original
scripts by calling the underlying step functions directly:
  1) gpt_extract_ae(...)
  2) filter_with_baseline(...)
  3) map_to_ctcae_medcpt(...)
  4) incremental_update.update_patient_history(...)

Expected local files (same folder or importable on PYTHONPATH):
  - ae_note_pipeline_update.py         (notes pipeline)
  - ae_lab_pipeline_update.py            (lab pipeline)
  - incremental_update.py

If your files have different names, update the imports below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import os
import pandas as pd

# ---- Import your two pipelines (module names must match your .py filenames) ----
import ae_note_pipeline_update as notes_mod
import ae_lab_pipeline_update as labs_mod

from incremental_update import update_patient_history


Mode = Literal["notes", "labs"]


@dataclass
class PipelineConfig:
    # Required
    ctcae_dict_csv: str
    medcpt_model_dir: str

    # Optional
    baseline_file: Optional[str] = None
    history_dir: Optional[str] = None  # if None => no incremental update
    similarity_threshold: Optional[float] = None  # default set by mode if None


def run_ae(
    mode: Mode,
    input_csv_path: str,
    output_csv_path: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Run the requested pipeline and return the merged_df (if history_dir provided),
    otherwise return the single-run final_df.

    Parameters
    ----------
    mode: "notes" or "labs"
    input_csv_path: path to notes CSV or lab CSV
    output_csv_path: where to save the per-run final_df CSV (before history merge)
    cfg: PipelineConfig

    Returns
    -------
    pd.DataFrame:
        - merged_df if history_dir is set
        - else final_df
    """
    if mode not in ("notes", "labs"):
        raise ValueError("mode must be 'notes' or 'labs'")

    # Normalize empty string -> None
    baseline_file = cfg.baseline_file if cfg.baseline_file not in ("", "None") else None
    history_dir = cfg.history_dir if cfg.history_dir not in ("", "None") else None

    if mode == "notes":
        # Step 1: GPT extract
        ae_df = notes_mod.gpt_extract_ae(input_csv_path)

        # Step 2: baseline (optional)
        ae_filtered = notes_mod.filter_with_baseline(ae_df, baseline_file)

        # Step 3: map
        final_df = notes_mod.map_to_ctcae_medcpt(
            ae_filtered,
            ctcae_dict_csv=cfg.ctcae_dict_csv,
            medcpt_model_dir=cfg.medcpt_model_dir,
        )

        # Notes pipeline does NOT do similarity filtering by default
        sim_thr = cfg.similarity_threshold

    else:
        # mode == "labs"
        ae_df = labs_mod.gpt_extract_ae(input_csv_path)
        ae_filtered = labs_mod.filter_with_baseline(ae_df, baseline_file)
        final_df = labs_mod.map_to_ctcae_medcpt(
            ae_filtered,
            ctcae_dict_csv=cfg.ctcae_dict_csv,
            medcpt_model_dir=cfg.medcpt_model_dir,
        )

        # Labs pipeline default similarity filtering is 0.9 in your script
        sim_thr = 0.9 if cfg.similarity_threshold is None else cfg.similarity_threshold

    # Optional similarity threshold filtering (works for either mode if provided)
    if sim_thr is not None and not final_df.empty and "Similarity_Top1" in final_df.columns:
        final_df = final_df.copy()
        final_df["Similarity_Top1"] = pd.to_numeric(final_df["Similarity_Top1"], errors="coerce")
        before_n = len(final_df)
        final_df = final_df[final_df["Similarity_Top1"] >= float(sim_thr)].copy()
        after_n = len(final_df)
        print(f"🔍 Similarity_Top1 filter: {before_n} -> {after_n} (>= {sim_thr})")

    # Save the per-run output (for reproducibility / debugging / download)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved per-run output -> {output_csv_path}")

    # Optional: incremental update
    if history_dir is None:
        return final_df

    merged_df = update_patient_history(
        ae_new_df=final_df,
        history_dir=history_dir,
        mrn_col="MRN",
    )
    return merged_df


def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="Run AE extraction pipeline (notes or labs) with a unified interface.")
    p.add_argument("--mode", required=True, choices=["notes", "labs"], help="Which pipeline to run.")
    p.add_argument("--input", required=True, help="Input CSV path (notes CSV or lab CSV).")
    p.add_argument("--output", required=True, help="Output CSV path for per-run final_df.")
    p.add_argument("--baseline", default="", help="Baseline Excel path (optional).")
    p.add_argument("--ctcae", required=True, help="CTCAE dictionary CSV path.")
    p.add_argument("--medcpt", required=True, help="Fine-tuned MedCPT model directory.")
    p.add_argument("--history-dir", default="", help="History directory for incremental update (optional).")
    p.add_argument("--sim-thr", default="", help="Similarity_Top1 threshold (optional). For labs default is 0.9.")
    p.add_argument("--save-merged", default="", help="If set, also save merged_df to this CSV path.")
    return p


def main():
    p = _build_argparser()
    args = p.parse_args()

    sim_thr = None
    if args.sim_thr not in ("", "None"):
        try:
            sim_thr = float(args.sim_thr)
        except ValueError:
            raise ValueError("--sim-thr must be a number, e.g. 0.9")

    cfg = PipelineConfig(
        ctcae_dict_csv=args.ctcae,
        medcpt_model_dir=args.medcpt,
        baseline_file=args.baseline,
        history_dir=args.history_dir if args.history_dir not in ("", "None") else None,
        similarity_threshold=sim_thr,
    )

    merged_or_final = run_ae(
        mode=args.mode,
        input_csv_path=args.input,
        output_csv_path=args.output,
        cfg=cfg,
    )

    if args.save_merged not in ("", "None"):
        os.makedirs(os.path.dirname(args.save_merged) or ".", exist_ok=True)
        merged_or_final.to_csv(args.save_merged, index=False)
        print(f"✅ Saved merged_df -> {args.save_merged}")


if __name__ == "__main__":
    main()

