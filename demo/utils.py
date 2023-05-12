import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


def generate_prompt(question, context, tokenizer, device):
    prompt = """You're an AI assistant that is designed to help people answer questions.

    Your name is `EnabledChat` and you are an open-source model available to everyone to use freely. The following is a conversation between a human and an AI assistant.

    You should use the following context to help you give correct and accurate answers.

    {context}

    Finish the following conversation in the same pattern using the context from before to generate accurate answer: 
    [|Human|]
    Hi!
    [|AI|]
    Hello. How can I help you today?
    [|Human|]
    {question}
    [|AI|]
    """

    inputs = tokenizer(
        [prompt.format(question=question, context=context)], return_tensors="pt"
    ).to(device)
    return inputs


if __name__ == "__main__":
    ...