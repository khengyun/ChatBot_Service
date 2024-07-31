import os
import re
import random
import subprocess
import gradio as gr

from backend.helpers import query_rag, initialize_chatbot
from backend.chatbot import ChatBot







# Initialize chatbot once
chatbotllm = ChatBot(model_id="llama3.1")

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))

    return history

def bot(history, message):
    gallery = None
    file_type = None
    file_path_list = message['files']
    user_prompt = message['text']
    # Retrieve relevant context
    context_text, retrieval_results  = query_rag(user_prompt)

    # Assuming get_response handles context and user prompt
    response_text = chatbotllm.get_response(context_text, user_prompt)
    
    # Update response in UI
    history += [[None, next(response_text)]]
    yield history, gr.MultimodalTextbox(value=None, interactive=True), gallery

    for response in response_text:
        history[-1][1] += response

        yield history, gr.MultimodalTextbox(value=None, interactive=True), gallery

    # Result: Response + sources
    sources = []
    names, links = [], []
    for doc, _score in retrieval_results:
        # Get id, link, name metadata
        id = doc.metadata.get("id", None)
        link = doc.metadata.get("link", None)
        name = doc.metadata.get("name", None)
        
        sources.append(id)
        if link != '' and name!= "":
            links += link.split(',')
            names += name.split(',')
    links = list(dict.fromkeys(links))
    names = list(dict.fromkeys(names))

    print(links)
    print(names)
    
    history[-1][1] += f"\n\nSources: {sources}"
    
    if len(names) > 0:
        gallery = [[link, title] for link, title in zip(links, names)]
        # history += [[
        #     None, 
        #     gallery
        # ]]
    
    yield history, gr.MultimodalTextbox(value=None, interactive=True), gallery

def fake_gan():
    images = [
        (random.choice(
            [
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
            ]
        ), f"label {i}")
        for i in range(5)
    ]
    return images

with gr.Blocks(title="K&K's Bot", fill_height=True) as demo:
    

    # Initialize chatbot interface
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    gallery = gr.Gallery(
        label="images", show_label=False, elem_id="gallery"
    , columns=[5], rows=[1], object_fit="contain", height="auto")

    # Initialize textbox
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )


    gr.Examples(
        examples=[
            {'text': "Thông tin cửa hàng", 'files': []},
            {'text': "Menu bữa sáng", 'files': []},
            {'text': "Menu trà chiều", 'files': []},
            {'text': "Menu bữa tối", 'files': []},
            {'text': "Những món bán chạy nhất!", 'files': []},
        ],
        inputs = chat_input,
    )
    
    # Press Enter to send message
    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], chatbot
    )

    bot_msg = chat_msg.then(
        bot, [chatbot, chat_input], [chatbot, chat_input, gallery], api_name="bot_response"
    )

    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    chatbot.like(print_like_dislike, None, None)

demo.queue()

if __name__ == "__main__":
    demo.launch(
        share=True
    )
