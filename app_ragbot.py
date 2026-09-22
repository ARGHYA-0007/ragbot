from fastapi import FastAPI
from ragbot_model import workflow,build_index,is_indexed
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, HTTPException
app = FastAPI()


@app.get('/')
def hello():
    return {'message':'hello you are using rag made by arghya'}

@app.post('/indexing')
def indexing(path):
    try:
        num_chunks = build_index(path)
        return {"status": "indexed", "chunks_created": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/query')
def call(query):
    if not is_indexed():
        raise HTTPException(status_code=400, detail="No PDF indexed yet — call /index-pdf first")
    result = workflow.invoke({'query':[HumanMessage(content=query)],'refined_retrieved_text':'','retry_count':0})
    return {"answer": result['answer']}
