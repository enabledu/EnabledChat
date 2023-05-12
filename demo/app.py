import sys
from threading import Thread

import gradio as gr
from transformers import TextIteratorStreamer

from utils import generate_prompt, load_model_and_tokenizer

if len(sys.argv) != 2:
    print("USAGE: python app.py <MODEL_NAME>")
    sys.exit(1)

model, tokenizer, device = load_model_and_tokenizer(model_name=sys.argv[1])
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

title = "**EnabledChat** v0.0"
description = "*Disclaimer:*"


def user(user_message, chatbot):
    return "", chatbot + [[user_message, None]]


def generate(chatbot, history):
    inputs = generate_prompt(question=chatbot[-1][0], context="", history=history, tokenizer=tokenizer, device=device)

    # Start a seperate thread to generate the answer
    generation_kwargs = dict(
        inputs, streamer=streamer, generation_config=model.generation_config
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output of the model
    chatbot[-1][1] = ""
    history = history + [[chatbot[-1][0], ""]]
    for new_text in streamer:
        chatbot[-1][1] += new_text
        history[-1][1] += new_text
        yield chatbot, history

    chatbot[-1][1] = chatbot[-1][1].strip("\n")
    history[-1][1] = history[-1][1].strip("\n")
    yield chatbot, history

def retry(chatbot, history):
    if len(chatbot) == 0:
        yield chatbot
        return

    _ = history.pop()
    chatbot[-1][1] = ""
    for x in generate(chatbot, history):
        yield x

def clear():
    return None, []

with gr.Blocks() as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot").style(height="100%")

    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Enter question").style(container=False)

    with gr.Row().style(equal_height=True):
        clear_btn = gr.Button("Clear")
        retry_btn = gr.Button("Regenerate")

    user_input.submit(
        user, [user_input, chatbot], [user_input, chatbot], queue=False
    ).then(generate, [chatbot, history], [chatbot, history])

    retry_btn.click(retry, [chatbot, history], [chatbot, history])
    clear_btn.click(clear, None, [chatbot, history], queue=False)


demo.title = "EnabledChat"

if __name__ == "__main__":
    demo.queue(concurrency_count=15).launch(share=True)
