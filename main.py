import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question, and answer naturally like you know about the question if relevent context is there, if no, don't answer
    <context>
    {context}
    <context>
    Question: {input}
    """
)
st.write("Init")
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="gemma:2b")

vectors = FAISS.load_local("INDIA_FAISS", embeddings, allow_dangerous_deserialization=True)
st.write("Ready")
# Streamlit UI
st.title("ChatGPT-style RAG Document Q&A")

# Display chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# User input
user_prompt = st.chat_input("Ask a question about the research paper")



if user_prompt:
    # Retrieve answer from LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start

    # Display response
    with st.chat_message("assistant"):
        st.write(response['answer'])
    
    # Save chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

    # Show document similarity results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
