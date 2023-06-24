"""
Gradio app for EnabledChat
"""
import re
import sys
from threading import Thread

import gradio as gr
from theme import custom_theme
from transformers import TextIteratorStreamer
from utils import load_model_and_tokenizer

TITLE = """<h2 align="center">🚀 EnabledChat demo</h2>"""
USER_NAME = "User"
BOT_NAME = "EnabledChat"

# DEFAULT_INSTRUCTIONS = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called EnabledChat, and a human user, called User. EnabledChat is a chatbot made by Mahmoud Hussein as part of a graduation project. In the following interactions, User and EnabledChat will converse in natural language, and EnabledChat will answer User's questions. EnabledChat was built to be respectful, polite and inclusive. EnabledChat will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins."""
RETRY_COMMAND = "/retry"
DEFAULT_INSTRUCTIONS = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called EnabledChat, and a human user, called User. EnabledChat is a chatbot made by a group of students at Zagazig University as part of a graduation project.
In the following interactions, User and EnabledChat will converse in natural language, and EnabledChat will answer User's questions. EnabledChat was built to be respectful, polite and inclusive. EnabledChat will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth.
The follwing is the history of the conversation:
User: Hi!
EnabledChat: Hello, how can I help you today?
"""

model_name = sys.argv[1] if len(sys.argv) > 1 else "0x70DA/EnabledChat-Falcon"
model, tokenizer, device = load_model_and_tokenizer(model_name=model_name)
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

regex = [r"User $", r"User: $", r"EnabledChat $", r"EnabledChat: $"]


def format_chat_prompt(
    message: str, chat_history, instructions: str = DEFAULT_INSTRUCTIONS
) -> str:
    instructions = instructions.strip(" ").strip("\n")
    prompt = ""
    history_len = 0
    for turn in reversed(chat_history):
        if history_len > 1500:
            _ = chat_history.pop(0)
            break
        user_message, bot_message = turn
        prompt = f"{USER_NAME}: {user_message}\n{BOT_NAME}: {bot_message}\n" + prompt
        history_len += len(user_message) + len(bot_message)

    prompt = instructions + f"\n{prompt}\n{USER_NAME}: {message}\n{BOT_NAME}: "
    prompt += f"""
Using the previous history, ONLY answer the following question from User
{USER_NAME}: {message}
{BOT_NAME}:"""
    print(prompt)
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

    with gr.Row(elem_id="button_container"):
        with gr.Column():
            retry_button = gr.Button("Regenrate")
        with gr.Column():
            delete_turn_button = gr.Button("Delete last turn")
        with gr.Column():
            clear_chat_button = gr.Button("Clear all history")

    def run_chat(
        message: str,
        chat_history,
        instructions: str = DEFAULT_INSTRUCTIONS,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if not message or (message == RETRY_COMMAND and len(chat_history) == 0):
            yield chat_history
            return

        if message == RETRY_COMMAND and chat_history:
            prev_turn = chat_history.pop(-1)
            user_message, _ = prev_turn
            message = user_message

        prompt = format_chat_prompt(message, chat_history, instructions)
        chat_history = chat_history + [[message, ""]]

        # Start a seperate thread to generate the answer
        inputs = tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        try:
            del inputs["token_type_ids"]
        except:
            pass

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=128,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        acc_text = ""
        for new_token in streamer:
            acc_text += new_token

            for reg in regex:
                acc_text = re.sub(reg, "", acc_text)

            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield chat_history
            acc_text = ""

    def delete_last_turn(chat_history):
        if chat_history:
            chat_history.pop(-1)
        return {chatbot: gr.update(value=chat_history)}

    def run_retry(
        message: str,
        chat_history,
        instructions: str = DEFAULT_INSTRUCTIONS,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        yield from run_chat(
            RETRY_COMMAND, chat_history, instructions, temperature, top_p
        )

    def clear_chat():
        return []

    inputs.submit(
        run_chat,
        [inputs, chatbot],
        outputs=[chatbot],
        show_progress=False,
    )
    inputs.submit(lambda: "", inputs=None, outputs=inputs)
    delete_turn_button.click(delete_last_turn, inputs=[chatbot], outputs=[chatbot])
    retry_button.click(
        run_retry,
        [inputs, chatbot],
        outputs=[chatbot],
        show_progress=False,
    )
    clear_chat_button.click(clear_chat, [], chatbot)


def get_demo():
    with gr.Blocks(
        css="""#chat_container {width: 700px; margin-left: auto; margin-right: auto;}
                #button_container {width: 700px; margin-left: auto; margin-right: auto;}
                #chatbot {font-size: 14px; min-height: 300px;}               
                """,
        theme=custom_theme,
    ) as demo:
        gr.HTML(TITLE)
        chat()

    return demo


if __name__ == "__main__":
    demo = get_demo()
    demo.queue(max_size=128, concurrency_count=16)
    demo.launch(share=True, show_api=False)
