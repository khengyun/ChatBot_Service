import os
import random
import subprocess
import gradio as gr

from scripts.chatbot_run import get_final_response
from utility.input_preprocess import get_file_type, extract_youtube_link, extract_img_link


color_map = {
    "harmful": "crimson",
    "neutral": "gray",
    "beneficial": "green",
}

def html_src(harm_level):
    return f"""
<div style="display: flex; gap: 5px;padding: 2px 4px;margin-top: -40px">
  <div style="background-color: {color_map[harm_level]}; padding: 2px; border-radius: 5px;">
  {harm_level}
  </div>
</div>
"""

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))

    return history

def bot(history, message):
    file_type = None
    file_path_list = message['files']
    user_prompt = message['text']

    response_source = get_final_response(query_text=user_prompt)
    history += [(None,response_source)]

    return history, gr.MultimodalTextbox(value=None, interactive=False)


with gr.Blocks(title="Koi's Bot", fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], chatbot
    )
    bot_msg = chat_msg.then(
        bot, [chatbot, chat_input], [chatbot, chat_input], api_name="bot_response"
    )
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    chatbot.like(print_like_dislike, None, None)

demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)