from typing import List, TypedDict
import time
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from typing import Union
from pydantic import Field
path = r"C:\Users\arghy\OneDrive\Desktop\3rd sem\1690629458.pdf"
docs = (PyPDFLoader(f'{path}').load())
# print(docs)
chunks = RecursiveCharacterTextSplitter(chunk_size = 900,chunk_overlap = 150).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

# result = retriever.invoke('what is the college name?')
# print("*"*50)
# print(result)
import re
def decompose_to_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
# strip_list = []
# count = 0
# for i in range(len(result)):
#     z=(decompose_to_sentences(result[i].page_content))
#     for j in range(len(z)):
#         strip_list.append(z[j])
# query = 'syllabus of 3rd semister'
class KeepOrDrop(BaseModel):
    keepordrop:bool = Field(description='return True if the strip is important to answer as per query and False if the sentence is not that important for the query')
llm = ChatOllama(model='qwen2.5:7b',temperature=0)
llm_keepordrop = llm.with_structured_output(KeepOrDrop)
while True:
    query = input('USER:')
    strip_list = []
    count = 0
    result = retriever.invoke(f'{query}')
    for i in range(len(result)):
        z=(decompose_to_sentences(result[i].page_content))
        for j in range(len(z)):
            strip_list.append(z[j])
    dropping_list = []
    for i in range(len(strip_list)):
        z = llm_keepordrop.invoke(f'query:{query},sentence:{strip_list[i]}')
        if not z.keepordrop:
            dropping_list.append(i)
    refined_text = ''
    for i in range(len(strip_list)):
        if i not in dropping_list:
            refined_text = refined_text + '\n' + strip_list[i]

    result = llm.invoke(f'query:{query} and retrieved text:{refined_text}')
    print('AI:',result.content)