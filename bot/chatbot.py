import os
from typing import List

from operator import itemgetter

import torch
from transformers import pipeline, BitsAndBytesConfig

from langchain_community.chat_models import ChatOllama
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

"""
This is a content from a user's input (image/video/pdf).
Content: {content}
"""
GENERAL_PROMPT_TEMPLATE = """
You are an AI assistant at K&K food store who always responds in Vietnamese! 
Your role is to support, answer customers' questions and suggest related foods at store.

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


class ChatBot:
    # Init chatbot from hugging face
    def __init__(self, enabled=None, model_id="meta-llama/Meta-Llama-3-8B") -> None:
        self.enabled = enabled
        
        self.model_id = model_id
        self.load_model()

        self.messages = []

    # Define prompt
    def get_prompt(self, context: str, question: str, information: str = None) -> List:
        prompt_template = GENERAL_PROMPT_TEMPLATE.format(context_str=context)

        self.messages = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt_template),
                MessagesPlaceholder(variable_name="human_message"),
            ]
        )
        self.user_prompt = HumanMessage(content=question)


    # Load model from transformer pipeline
    def load_model(self):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="float16",
        #     bnb_4bit_use_double_quant=True
        # )
        # self.llm = HuggingFacePipeline.from_model_id(
        #     model_id=self.model_id,
        #     task="text-generation",
        #     pipeline_kwargs=dict(
        #         max_new_tokens=512, 
        #         do_sample=True,
        #         temperature = 0.3,
        #         top_p = 0.9,
        #         repetition_penalty=1.03,
        #     ),
        #     model_kwargs={"quantization_config": quantization_config,
        #                   "low_cpu_mem_usage": True},
        # )

        # self.chat_model = ChatHuggingFace(llm=self.llm)
        self.chat_model = ChatOllama(model=self.model_id, temperature=0.2, top_k=10, top_p=0.5)

        self.trimmer = trim_messages(
            max_tokens=20000,
            strategy="last",
            token_counter=self.chat_model,
            include_system=True,
            allow_partial=True,
        )

    # Get response from chatbot
    def get_response(self):
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

        for r in with_message_history.stream(
            {"human_message": [self.user_prompt]},
            config=config,
        ):
            yield r.content

        # response = with_message_history.invoke(
        #     {"human_message": [self.user_prompt]},
        #     config=config,
        # )
        store[session_id].messages = self.trimmer.invoke(store[session_id].messages)
        # return response.content