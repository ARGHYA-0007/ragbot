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
# @app.post('/indexing')
# def indexing(path):
#     try:
#         num_chunks = build_index(path)
#         return {"status": "indexed", "chunks_created": num_chunks}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.pdf', '.csv', '.txt', '.docx', '.xlsx', '.xls'}

@app.post('/indexing')
async def indexing(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        num_chunks = build_index(save_path)
        return {"status": "indexed", "chunks_created": num_chunks, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/query')
def call(query):
    result = workflow.invoke(
        {'query':[HumanMessage(content=query)], 'refined_retrieved_text':'', 'retry_count':0, 'ai_query':''},
        config=config
    )
    return {"answer": result['query'][-1].content}
