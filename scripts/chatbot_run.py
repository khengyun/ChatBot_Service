import os
import argparse

from typing import List

from backend.query_data import query_rag
from bot import ChatBot

def read_des(desc_path: str = "data/desc_summ.txt"):
    enabled, information = None, None
    if os.path.exists(desc_path):
        with open(desc_path, "r") as f:
            desc_summ = f.readlines()

        # Remove desc pạth
        os.remove(desc_path)

        enabled = desc_summ[0].strip("\n")
        information = ''.join(desc_summ[1:])

    return enabled, information

def get_final_response(query_text: str):
    # Read image description/ video summarization
    enabled, information = read_des()

    # Retrieve relevant context
    context_text, retrieval_results  = query_rag(query_text)

    # Initialize chatbot, input prompt and get response
    model = ChatBot(enabled=enabled, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
    model.get_prompt(
            context_text, 
            query_text, 
            information=information
            )
    response_text = model.get_response()

    # Result: Response + sources
    info_sources = []
    sources = [doc.metadata.get("id", None) for doc, _score in retrieval_results]
    sources += info_sources

    if len(sources):
        formatted_response = f"{response_text}\n\nSources: {sources}"
    else:
        formatted_response = f"{response_text}"

    return formatted_response

if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    formatted_response = get_final_response(query_text=query_text)
    # Save result
    save_dir = "response"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "final_response.txt")
    with open(save_path, 'w') as f:
        f.write(formatted_response)