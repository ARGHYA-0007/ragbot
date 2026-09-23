from fastapi import FastAPI, UploadFile, File, HTTPException
from ragbot_model import workflow, build_index, is_indexed
from langchain_core.messages import HumanMessage
import os
import shutil

app = FastAPI()
config = {"configurable": {"thread_id": '1'}}

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
    result = workflow.invoke(
        {'query':[HumanMessage(content=query)], 'refined_retrieved_text':'', 'retry_count':0, 'ai_query':''},
        config=config
    )
    return {"answer": result['query'][-1].content}
