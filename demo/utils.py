import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


if __name__ == "__main__":
    ...
