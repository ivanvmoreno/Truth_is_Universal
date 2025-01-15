import os
from typing import Tuple

import configparser
import pandas as pd
import sklearn.metrics as metrics
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_prompt(
    subject, question, answers, sys_prompt: callable, choices=["A", "B", "C", "D"]
) -> Tuple[str, str]:
    user_prompt = (
        f"{question}\n"
        + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        + "\nAnswer:"
    )

    return f"{SYS_PROMPT(subject)}\n{user_prompt}"


def format_dataset(base_path: str) -> pd.DataFrame:
    df = pd.read_parquet(base_path)
    prompts = df.apply(
        lambda row: format_prompt(
            row["subject"], row["question"], row["choices"], SYS_PROMPT
        ),
        axis=1,
    )
    answers = df["answer"].apply(lambda a: ANSWER_MAP[a])
    formatted = pd.DataFrame(
        {"prompt": prompts.tolist(), "answer": answers, "subject": df["subject"]}
    )
    return formatted


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    for d in DATASET_CATS:
        dataset_f = format_dataset(f"{DATASETS_DIR}/{DATASET_NAME}/{d}")
        dataset_f.to_csv(
            f"{DATASETS_DIR}/{FORMATTED_DATASET_NAME}/{d}.csv", index=False
        )
