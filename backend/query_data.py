import os
import argparse

from typing import List

from langchain_community.vectorstores import Chroma
from utility import get_embedding_function

CHROMA_PATH = "food_database3"

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

    print(results)
    return context_text, results

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    context = query_rag(query_text)

if __name__ == "__main__":
    main()
