import os
import time
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone.vectorstores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = openai_api_key


# Initialize Pinecone client
def initialize_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(pinecone_index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(pinecone_index_name)


# Load documents
def load_document(file_path: str, file_type: str):
    if file_type == "application/pdf":
        return PyPDFLoader(file_path).load_and_split()
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return Docx2txtLoader(file_path).load()
    elif file_type == "text/plain":
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")


# Split documents into chunks
def split_document(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


# Embed user's query
def get_embedding(text: str) -> list:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = embeddings.embed_query(text)
    return embedding


# Embed documents and store in Pinecone
def embed_and_store_document(docs, index_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    vectorstore.add_documents(docs)


def add_texts_to_index(texts, index_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    vectorstore.add_texts(texts)
