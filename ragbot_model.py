from typing import List, TypedDict
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
path = r"C:\Users\arghy\OneDrive\Desktop\3rd sem\1690629458.pdf"
docs = (PyPDFLoader(f'{path}').load())
# print(docs)
chunks = RecursiveCharacterTextSplitter(chunk_size = 900,chunk_overlap = 150).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

result = retriever.invoke('what is the college name?')
print("*"*50)
print(result)