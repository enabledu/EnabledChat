"""
Utility functions for the app
"""

import torch
from transformers import AutoTokenier, AutoModelForCasualLM


def load_model_and_tokenizer(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCasualLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    )
    return model, tokenizer, device
