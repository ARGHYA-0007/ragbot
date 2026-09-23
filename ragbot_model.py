from typing import List, TypedDict,Annotated
import time
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langgraph.graph.message import add_messages, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from typing import Union,Literal
from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.checkpoint.memory import MemorySaver
import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
import os

def get_loader(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return PyPDFLoader(path)
    elif ext == '.csv':
        return CSVLoader(path)
    elif ext == '.txt':
        return TextLoader(path, encoding='utf-8')
    elif ext == '.docx':
        return Docx2txtLoader(path)
    elif ext in ('.xlsx', '.xls'):
        return UnstructuredExcelLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
retrieve_again = '''You are a strict retrieval quality evaluator.

Your job is to determine whether the retrieved documents are sufficient
and correctly matched to the user's query.

Return "yes" if another retrieval is necessary.

Return "yes" when:
- The retrieved documents do not contain enough information.
- The retrieved documents contain information from the wrong semester.
- The retrieved documents contain information from the wrong branch.
- Important subjects or requested fields are missing.
- A more targeted search could improve accuracy.

Return "no" ONLY when:
- The retrieved information directly matches the requested semester/branch/topic.
- The important requested information is present.
- The retrieved documents are sufficient to answer accurately.
- can be answer the query using the retrieved text

For queries asking for ALL subjects, verify that the retrieved
information appears complete rather than assuming that a few subjects
are sufficient.

Return only "yes" or "no".
'''
sentence_filter_prompt = '''You are a lenient relevance filter.

You will be given a query and a numbered list of sentences.

For each sentence, return keep=True if the sentence has ANY relevant or overlapping context with the query — 
even partial, indirect, or loosely related information counts as True.

Return keep=False ONLY if the sentence is about a totally different, unrelated topic 
with no connection to the query at all.

When in doubt, return True. Only drop sentences that are clearly and completely irrelevant.

Return a decision for every sentence, using its index.'''
class SentenceDecision(BaseModel):
    index: int = Field(description='the index of the sentence')
    keep: bool = Field(description='True if sentence has any relevant/overlapping context with the query, False only if totally unrelated')

class KeepOrDropBatch(BaseModel):
    decisions: List[SentenceDecision]
# load_dotenv()
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash"
# )
llm = ChatOllama(model='qwen2.5:7b',temperature=0)
# llm_keepordrop = llm.with_structured_output(KeepOrDrop)
llm_keepordrop_batch = llm.with_structured_output(KeepOrDropBatch)

class again_retrieve(BaseModel):
    enoughornot: Literal['yes','no'] = Field(description=f'{retrieve_again}')
llm_retrieve_again = llm.with_structured_output(again_retrieve)

retriever = None
vector_store = None
def is_indexed():
    return retriever is not None
def build_index(path: str):
    global retriever, vector_store

    docs = PyPDFLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75).split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    return len(chunks)
class isitchat(BaseModel):
    isitnormal:Literal['yes','no'] = Field(description='''You are a query classifier for a RAG chatbot.

Your task is to determine whether the user's query can be answered as normal conversation or whether it requires retrieving information from the provided documents.

Return **"yes"** if the query is normal conversation and does NOT require document retrieval.

Return **"no"** if the query requires information from the documents and should be sent to the retrieval system.

Examples:

* "Hi" → yes
* "How are you?" → yes
* "Tell me a joke" → yes
* "What is the capital of India?" → yes
* "What subjects are in the 3rd semester IT syllabus?" → no
* "Explain the topic mentioned in the PDF" → no
* "What does the document say about attendance?" → no
* "According to the PDF, what is the eligibility criteria?" → no

Return only **"yes"** or **"no"**. Do not provide any explanation.
''')
llm_isnormal = llm.with_structured_output(isitchat)
class RagState(TypedDict):
    query:Annotated[list[BaseMessage], add_messages]
    ai_query:str
    refined_retrieved_text:str
    answer:str
    retrieve_again:str
    retry_count:int
    normal_chat:str
def chat(state:RagState):
    yah_no = llm_isnormal.invoke(state['query'][-1].content)
    return {'normal_chat':yah_no.isitnormal}
def route_chat(state:RagState):
    if state['normal_chat'].lower() == 'yes':
        # 
        return 'chatting'
    elif state['normal_chat'].lower() == 'no':
        return 'retrieve'
def chatting(state:RagState):
    result = llm.invoke(state['query'])
    return {'query':[result]}
def retrieve_to_refine(state:RagState):
    if state['ai_query']:
        query = state['ai_query']
    else:
        query = state['query'][-1].content
    # strip_list = []
    previous_text = state['refined_retrieved_text']
    result = retriever.invoke(f'{query}')
    refined_text = '\n'.join(
    doc.page_content for doc in result
)
    # for i in range(len(result)):
    #     z = decompose_to_sentences(result[i].page_content)
    #     for j in range(len(z)):
    #         strip_list.append(z[j])

    # numbered = '\n'.join(f'{i}: {s}' for i, s in enumerate(strip_list))
    # batch_result = llm_keepordrop_batch.invoke(
    #     f'{sentence_filter_prompt}\n\nquery: {query}\n\nsentences:\n{numbered}'
    # )

    # keep_indices = {d.index for d in batch_result.decisions if d.keep}
    # refined_text = '\n'.join(strip_list[i] for i in keep_indices if i < len(strip_list))

    total = previous_text + '\n' + refined_text
    return {'refined_retrieved_text': total, 'retry_count': state.get('retry_count',0) + 1,}
def enough(state:RagState):
    query =  state['query'][-1].content
    text = state['refined_retrieved_text']
    yes_or_no = llm_retrieve_again.invoke(f'query:{query} and retrieved docs are:\n {text}')
    return {'retrieve_again':yes_or_no.enoughornot}

def route(state:RagState):
    if state['retrieve_again'].lower() == 'yes' and state.get('retry_count',0) < 3:
        return 'new_query'
    elif state['retrieve_again'].lower() == 'no':
        return 'generate'
    else:
        return 'generate'

def new_query(state:RagState):
    new_query = llm.invoke(f"""
You are a query rewriting agent for a PDF retrieval system.

The human has provided the following query:

{state["query"][-1].content} and query made by ai are ({state['ai_query']}) and retrieved data are ({state['refined_retrieved_text']})

Rewrite this query into a new, more precise search query that can retrieve additional relevant information from the PDF.

Requirements:
- Preserve the original intent of the human's query.
- Identify the key concepts, entities, keywords, and context.
- Add useful related terms that may appear in the PDF.
- Make the query more specific and retrieval-friendly.
- You can make new query related to the first query to answer the question 
- Do not answer the question.
- Return only the rewritten query, with no explanation.
""")
    return {'ai_query':new_query.content}
def generate(state:RagState):
    result = llm.invoke(f'''Answer the query using ONLY the retrieved documents below. query: {state["query"][-1].content}
retrieved documents: {state["refined_retrieved_text"]}''').content
    return {'query': [result]}
def decompose_to_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
graph = StateGraph(RagState)
graph.add_node('chat',chat)
graph.add_node('chatting',chatting)
graph.add_node('retrieve',retrieve_to_refine)
graph.add_node('enough',enough)
graph.add_node('generate',generate)
graph.add_node('new_query',new_query)

graph.add_edge(START,'chat')
graph.add_conditional_edges('chat',route_chat)
graph.add_edge('chatting',END)
graph.add_edge('retrieve','enough')
graph.add_conditional_edges('enough',route)
graph.add_edge('new_query','retrieve')
graph.add_edge('generate',END)
checkpoint = MemorySaver()
workflow = graph.compile(checkpointer=checkpoint)
# while True:
#     query = input('USER:')
#     if query.lower() in ['exit','bye','stop']:
#         break
#     result = workflow.invoke({'query':[HumanMessage(content=query)],'refined_retrieved_text':'','retry_count':0})
#     print('AI:',result['answer'])