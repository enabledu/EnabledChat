"""
Gradio app for EnabledChat
"""
import sys
from threading import Thread
import gradio as gr

# from transformers import TextIteratorStreamer

# from utils import load_model_and_tokenizer

TITLE = """<h2 align="center">🚀 EnabledChat demo</h2>"""
USER_NAME = "User"
BOT_NAME = "EnabledChat"

DEFAULT_INSTRUCTIONS = f"""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called EnabledChat, and a human user, called User. EnabledChat is a chatbot made by Mahmoud Hussein as part of a graduation project. In the following interactions, User and EnabledChat will converse in natural language, and EnabledChat will answer User's questions. EnabledChat was built to be respectful, polite and inclusive. EnabledChat will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
"""
RETRY_COMMAND = "/retry"

model_name = sys.argv[1] if len(sys.argv) > 1 else "0x70DA/EnabledChat-Falcon"
# model, tokenizer, device = load_model_and_tokenizer(model_name=model_name)
# streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)


def format_chat_prompt(message: str, chat_history, instructions: str) -> str:
    instructions = instructions.strip(" ").strip("\n")
    prompt = instructions
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\n{USER_NAME}: {user_message}\n{BOT_NAME}: {bot_message}"
    prompt = f"{prompt}\n{USER_NAME}: {message}\n{BOT_NAME}:"
    return prompt


def chat():
    with gr.Column(elem_id="chat_container"):
        with gr.Row():
            chatbot = gr.Chatbot(elem_id="chatbot")
        with gr.Row():
            with gr.Column(elem_id="input"):
                inputs = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    max_lines=3,
                )
            with gr.Column(elem_id="send_button"):
                send = gr.Button(value="Send")

    with gr.Row(elem_id="button_container"):
        with gr.Column():
            retry_button = gr.Button("♻️ Retry last turn")
        with gr.Column():
            delete_turn_button = gr.Button("🧽 Delete last turn")
        with gr.Column():
            clear_chat_button = gr.Button("✨ Delete all history")


def get_demo():
    with gr.Blocks(
        css="""#chat_container {width: 700px; margin-left: auto; margin-right: auto;}
                #button_container {width: 700px; margin-left: auto; margin-right: auto;}
                #chatbot {font-size: 14px; min-height: 300px;}               
                """
    ) as demo:
        gr.HTML(TITLE)
        chat()

    return demo


if __name__ == "__main__":
    demo = get_demo()
    demo.queue(max_size=128, concurrency_count=16)
    demo.launch(share=False)
