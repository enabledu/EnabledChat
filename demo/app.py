import sys

import gradio as gr
from transformers import TextIteratorStreamer

from utils import load_model_and_tokenizer

if len(sys.argv) != 2:
    print("USAGE: python app.py <MODEL_NAME>")
    sys.exit(1)

model, tokenizer, device = load_model_and_tokenizer(model_name=sys.argv[1])

title = "**EnabledChat** v0.0"
description = "*Disclaimer:*"


def user(user_message, history):
    if user_message == "":
        return None, None
    return "", history + [[user_message, None]]


def generate(chatbot):
    ...


def retry(chatbot):
    if len(chatbot) == 0:
        yield chatbot
        return

    chatbot[-1][1] = ""
    for x in generate(chatbot):
        yield x


with gr.Blocks() as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot").style(height="100%")

    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Enter question")

    with gr.Row().style(equal_height=True):
        clear_btn = gr.Button("Clear")
        retry_btn = gr.Button("Regenerate")

    user_input.submit(
        user, [user_input, chatbot], [user_input, chatbot], queue=False
    ).then(generate, chatbot, chatbot)
    retry_btn.click(retry, chatbot, chatbot)
    clear_btn.click(lambda: None, None, chatbot, queue=False)


demo.title = "EnabledChat"

if __name__ == "__main__":
    demo.queue().launch()
