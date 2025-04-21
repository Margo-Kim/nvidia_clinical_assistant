# evaluation.py  – no other modules touched
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm   # pip install tqdm, if missing

log = logging.getLogger(__name__)


def load_fiqa_eval(limit=None):
    # ask HF datasets to give the split directly
    ds = load_dataset(
        "explodinggradients/fiqa",
        "ragas_eval",
        split="baseline"    
    )
    if limit:
        ds = ds.select(range(limit))

    
    df = pd.DataFrame({
        "question":      ds["question"],
        "ground_truths": ds["ground_truths"]  # or ds["answer"] depending on column name
    })
    return df

def fill_answers(rag_sys, df: pd.DataFrame, save_every=10,
                 out_path: str | Path = "fiqa_answers.csv") -> None:
    """
    Runs rag_sys.query() on each question and writes a CSV:
        question, ground_truths, answer, source_texts
    """
    df = df.copy()
    df["answer"]        = None
    df["contexts"]  = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        try:
            resp = rag_sys.query(row["question"])
            df.at[idx, "answer"] = resp.get("answer", "")
            df.at[idx, "contexts"] = " | ".join(
                doc.page_content[:120].replace("\n", " ")
                for doc in resp.get("context", [])
            )
        except Exception as e:
            df.at[idx, "answer"] = f"ERROR: {e}"

        if (idx + 1) % save_every == 0:
            df.to_csv(out_path, index=False)

    df.to_csv(out_path, index=False)
    log.info("Answers written ➜ %s", out_path)
