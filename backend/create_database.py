import re, os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma


from helpers import get_embedding_function

from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")

def calculate_chunk_ids(chunks):
    '''
    This will create IDs like "data/monopoly.md:2"
    Page Source : Chunk Index
    '''

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        current_page_id = source

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
    
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(), collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("No new documents to add")

def extract_link(chunk_text):
    # Extract and remove links
    name_pattern = re.compile(r'"name":\s*"([^"]+)"')
    link_pattern = re.compile(r'"links":\s*([\w/\.]+)')
    link = link_pattern.findall(chunk_text)
    name = name_pattern.findall(chunk_text)

    # Remove the links from the chunk_text
    chunk_text_without_link = link_pattern.sub('', chunk_text)

    chunk_text_without_link = re.sub(r'\s*(\n|$)', '', chunk_text_without_link)

    return ','.join(name), ',data/'.join(link), chunk_text_without_link

def split_text(documents: list[Document]):
    text_splitter = CharacterTextSplitter(separator=',\n\n', chunk_size=250, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    for chunk in chunks:
        name, link, new_content = extract_link(chunk.page_content)
        chunk.page_content = new_content
        
        # Assign new metadata
        chunk.metadata['name'] = name
        chunk.metadata['link'] = 'data/' + link
        
    return chunks


def load_documents():
    # Load documents from datapath
    documents_loader = DirectoryLoader(DATA_PATH, glob="*/*.md")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH, extract_images=True)
    documents = documents_loader.load()
    pdfs = pdf_loader.load()
    final_documents = documents + pdfs
    
    return final_documents


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    
    add_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()