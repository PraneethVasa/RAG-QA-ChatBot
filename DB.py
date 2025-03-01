from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# Creates vector embeddings and saves it in the FAISS DB
faiss_db = FAISS.from_documents(texts, embedings)

# Saves and export the vector embeddings database
faiss_db.save_local("ipc_vector_db")