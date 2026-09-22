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
import re
retrieve_again = '''You are a strict retrieval quality evaluator.

Your job is to determine whether the retrieved documents are sufficient
and correctly matched to the user's query.

Return "yes" if another retrieval is necessary.

Return "yes" when:
- The retrieved documents do not contain enough information.
- The retrieved documents contain information from the wrong semester.
- The retrieved documents contain information from the wrong branch.
- Important subjects or requested fields are missing.
- The retrieved documents contain mixed or conflicting information.
- A more targeted search could improve accuracy.

Return "no" ONLY when:
- The retrieved information directly matches the requested semester/branch/topic.
- The important requested information is present.
- There is no obvious irrelevant or conflicting semester information.
- The retrieved documents are sufficient to answer accurately.

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
llm = ChatOllama(model='qwen2.5:7b',temperature=0)
# llm_keepordrop = llm.with_structured_output(KeepOrDrop)
llm_keepordrop_batch = llm.with_structured_output(KeepOrDropBatch)

class enough_or_not(BaseModel):
    enoughornot: Literal['yes','no'] = Field(description=f'{retrieve_again}')
llm_retrieve_again = llm.with_structured_output(enough_or_not)



path = r"C:\Users\arghy\OneDrive\Desktop\3rd sem\1690629458.pdf"
docs = (PyPDFLoader(f'{path}').load())
# print(docs)
chunks = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 75).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':10})
class RagState(TypedDict):
    query:Annotated[list, add_messages]
    refined_retrieved_text:str
    answer:str
    retrieve_again:str
    retry_count:int

def retrieve_to_refine(state:RagState):
    query = state['query'][-1].content
    strip_list = []
    previous_text = state['refined_retrieved_text']
    result = retriever.invoke(f'{query}')
#     refined_text = '\n'.join(
#     doc.page_content for doc in result
# )
    for i in range(len(result)):
        z = decompose_to_sentences(result[i].page_content)
        for j in range(len(z)):
            strip_list.append(z[j])

    numbered = '\n'.join(f'{i}: {s}' for i, s in enumerate(strip_list))
    batch_result = llm_keepordrop_batch.invoke(
        f'{sentence_filter_prompt}\n\nquery: {query}\n\nsentences:\n{numbered}'
    )

    keep_indices = {d.index for d in batch_result.decisions if d.keep}
    refined_text = '\n'.join(strip_list[i] for i in keep_indices if i < len(strip_list))

    total = previous_text + '\n' + refined_text
    return {'refined_retrieved_text': total, 'retry_count': state.get('retry_count',0) + 1}
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

{state["query"]}

Rewrite this query into a new, more precise search query that can retrieve additional relevant information from the PDF.

Requirements:
- Preserve the original intent of the human's query.
- Identify the key concepts, entities, keywords, and context.
- Add useful related terms that may appear in the PDF.
- Make the query more specific and retrieval-friendly.
- Do not change the meaning of the original query.
- Do not answer the question.
- Return only the rewritten query, with no explanation.
""")
    return {'query':HumanMessage(content=new_query.content)}
def generate(state:RagState):
    result = llm.invoke(f'generate answer as per the query:{state["query"][0]} and the retrieve documents:{state["refined_retrieved_text"]}').content
    return {'answer':result}
def decompose_to_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
graph = StateGraph(RagState)
graph.add_node('retrieve',retrieve_to_refine)
graph.add_node('enough',enough)
graph.add_node('generate',generate)
graph.add_node('new_query',new_query)

graph.add_edge(START,'retrieve')
graph.add_edge('retrieve','enough')
graph.add_conditional_edges('enough',route)
graph.add_edge('new_query','retrieve')
graph.add_edge('generate',END)
workflow = graph.compile()
while True:
    query = input('USER:')
    if query.lower() in ['exit','bye','stop']:
        break
    result = workflow.invoke({'query':[HumanMessage(content=query)],'refined_retrieved_text':'','retry_count':0})
    print('AI:',result['answer'])