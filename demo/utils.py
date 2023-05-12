import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


def generate_prompt(question, context, history, tokenizer, device):
    prompt = """You're an AI assistant that is designed to help people answer questions.

Your name is `EnabledChat` and you are an open-source model available to everyone to use freely. The following is a conversation between a human and an AI assistant.

You should use the following context to help you give correct and accurate answers.

{context}

Finish the following conversation in the same pattern using the context from before to generate accurate answer: 
[|Human|]
Hi!
[|AI|]
Hello. How can I help you today?{history}
{question}
[|AI|]
"""

    question = "[|Human|]\n" + question
    prompt_history = ""
    history_len = 0
    for human, ai in reversed(history):
        if history_len > 1500:
            _ = history.pop(0)
            break
        prompt_history = f"\n[|Human|]\n{human}\n[|AI|]\n{ai}" + prompt_history
        history_len += len(human.split()) + len(ai.split())

    inputs = tokenizer(
        [prompt.format(question=question, context=context, history=prompt_history)],
        return_tensors="pt",
    ).to(device)
    return inputs


if __name__ == "__main__":
    ...
