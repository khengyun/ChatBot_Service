import os
from dotenv import load_dotenv

from typing import List

import yt_dlp
from PIL import Image
from utility import read_img
from operator import itemgetter

import torch
import whisper
from transformers import pipeline, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GENERAL_PROMPT_TEMPLATE = """
You are an AI assistant at a Vietnamese food store who always responds in Vietnamese! 
Your role is to support, answer customers' questions and suggest related foods at store.

This is a content from a user's input (image/video/pdf).
Content: {content}

Below are some relevant contexts of a question from a user. 
Contexts: {context_str}

Answer the question given the information in those contexts and the content. 
If relevant contexts are None, just answer normally.
If you cannot find the answer to the question, say "I don't know" in Vietnamese.
"""
store = {}
config = {"configurable": {"session_id": "koi1"}}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]


class MultiModalBot:
    def __init__(self, path: str, llava_model_id = "llava-hf/llava-v1.6-mistral-7b-hf", whisper_model_id = 'base') -> None:
        self.path = path
        self.vid_path = None
        self.llava_model_id = llava_model_id
        self.whisper_model_id = whisper_model_id

        self.get_model()

    def get_model(self):
        # Initialize image-enabled llm
        self.llm_img_processor = LlavaNextProcessor.from_pretrained(self.llava_model_id)

        self.llm_img = LlavaNextForConditionalGeneration.from_pretrained(
            self.llava_model_id, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        ) 

        # Initialize audio-enabled llm
        self.llm_audio = whisper.load_model(self.whisper_model_id)

    def download_mp4_from_youtube(self):
        # Set the options for the download
        name = self.path.split('/')[-1] 
        filename = f'data/video/{name}.mp4'
        self.vid_path = filename
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': filename,
            'quiet': True,
        }
        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(self.path, download=True)

    def describe(self, prompt):
        # image_b64 = convert_to_base64(self.path)
        image = read_img(self.path)

        # llava_prompt = f"[INST] <image>\n{prompt} [/INST]"
        llava_prompt = f"[INST] <image>\nDescribe the image in details, and provide relevant information if possible [/INST]"
        inputs = self.llm_img_processor(llava_prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = self.llm_img.generate(**inputs, max_new_tokens=512)

        description = self.llm_img_processor.decode(output[0], skip_special_tokens=True)

        return description.split('[/INST]')[1]
    
    def summarize(self):
        if self.vid_path != None:
            result = self.llm_audio.transcribe(self.vid_path)
        else:
            result = self.llm_audio.transcribe(self.path)

        return result['text']
    
class ChatBot:
    # Init chatbot from hugging face
    def __init__(self, enabled=None, model_id="meta-llama/Meta-Llama-3-8B") -> None:
        self.enabled = enabled
        
        self.model_id = model_id
        self.load_model()

        self.messages = []

    # Define prompt
    def get_prompt(self, context: str, question: str, information: str = None) -> List:
        prompt_template = GENERAL_PROMPT_TEMPLATE.format(context_str=context, content=information)

        self.messages = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt_template),
                MessagesPlaceholder(variable_name="human_message"),
            ]
        )
        self.user_prompt = HumanMessage(content=question)


    # Load model from transformer pipeline
    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_id,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=True,
                temperature = 0.5,
                top_p = 0.9,
                repetition_penalty=1.03,
            ),
            model_kwargs={"quantization_config": quantization_config,
                          "low_cpu_mem_usage": True},
        )

        self.chat_model = ChatHuggingFace(llm=self.llm)

        self.trimmer = trim_messages(
            max_tokens=8000,
            strategy="last",
            token_counter=self.chat_model,
            include_system=True,
            allow_partial=True,
        )

    # Get response from chatbot
    def get_response(self) -> str:
        session_id = config['configurable']['session_id']
        chain = (
            RunnablePassthrough.assign(messages=itemgetter("human_message") | self.trimmer)
            | self.messages 
            | self.chat_model
        )
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="human_message",
        )
        response = with_message_history.invoke(
            {"human_message": [self.user_prompt]},
            config=config,
        )
        print(store[session_id].messages)
        store[session_id].messages = self.trimmer.invoke(store[session_id].messages)
        print(store[session_id].messages)

        return response.content.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1]