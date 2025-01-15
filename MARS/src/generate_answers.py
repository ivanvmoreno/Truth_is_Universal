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
