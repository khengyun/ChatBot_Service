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
from backend.prompt.prompts import GENERAL_PROMPT_TEMPLATE

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

        self.chat_model = ChatOllama(model=self.model_id, temperature=0.2, top_k=10, top_p=0.5)

        self.trimmer = trim_messages(
            max_tokens=100000,
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

        store[session_id].messages = self.trimmer.invoke(store[session_id].messages)