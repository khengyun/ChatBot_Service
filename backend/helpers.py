import torch 

from backend import ChatBot
from langchain_huggingface import HuggingFaceEmbeddings

import os
import argparse

from typing import List

from langchain_chroma import Chroma


from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")


def initialize_chatbot():
    # Initialize chatbot, input prompt and get response
    model = ChatBot(model_id="llama3.1")
    
    return model

def get_embedding_function():
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "google-bert/bert-base-multilingual-cased"
    # model_name = "BAAI/bge-m3"
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embedding_function

def query_rag(query_text: str) -> str:
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    context_text = ""
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5, score_threshold=0.4)
    if len(results) == 0:
        print(f"Unable to find matching results. Continuing to use the bot's knowledge.")
        results = []
        # return
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    return context_text, results
