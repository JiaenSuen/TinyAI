import hashlib
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from nlp_functions import get_embedding_function
import os

class Params:
    DATA_PATH   = "./PDFs"
    CHROMA_PATH = "./chroma_langchain_db"




def init_vector_database():
    if not os.path.exists(Params.CHROMA_PATH) or not os.listdir(Params.CHROMA_PATH):
        print("Chroma vector database not found, building a new one...")
        build_chroma_index()
    else:
        print("Chroma vector database already exists")

 
def load_documents():
    loader = PyPDFDirectoryLoader(Params.DATA_PATH)
    documents = loader.load()
    print(f"Success load {len(documents)} page PDF files")
    return documents


 
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(chunks)} texts")
    return chunks


# Chroma Database
def build_chroma_index():
    documents = load_documents()
    chunks    = split_documents(documents)

    db = Chroma(
        persist_directory=Params.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    new_chunk_ids = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    db.add_documents(chunks, ids=new_chunk_ids)
    print(f"Chroma vector database has been build, total {len(chunks)} data amount")


# Retriver
def get_retriever():
    db = Chroma(
        persist_directory=Params.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    return db.as_retriever()
