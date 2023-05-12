import gradio as gr
import random
import time

title = "**EnabledChat** v0.0"
description = "*Disclaimer:*"

def user(user_message, history):
    return "", history + [[user_message, None]]

def generate(history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

def retry(history):
    if len(history) == 0:
        yield history
        return

    history[-1][1] = ""
    for x in generate(history):
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

    user_input.submit(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        generate, chatbot, chatbot
    )
    retry_btn.click(retry, chatbot, chatbot)
    clear_btn.click(lambda: None, None, chatbot, queue=False)


demo.title = "EnabledChat"
demo.queue().launch()
