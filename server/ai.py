from os import getenv
from huggingface_hub import InferenceClient
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
import numpy
import faiss

"""
imports for chromadb instead of FAISS for vector store

from langchain_chroma import Chroma
import chromadb
"""

# setup huggingface serverless api
# note: in the future, should probably be setup as an inference model (but this might cost money)
HF_API_KEY = getenv("HF_API_KEY")

# note: Vision-Instruct broke due to gradio update, might be fixed by the time you read this
# for rag question answering
# GEN_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
GEN_MODEL = "meta-llama/Llama-3.2-3B-Instruct" # "meta-llama/Llama-3.2-11B-Vision-Instruct"

FE_MODEL = "thenlper/gte-large" # for feature extraction
QA_MODEL = "deepset/roberta-base-squad2" # for non-rag question answering (mostly depreciated, mostly used for pinging)
SCORE_THRESHOLD = 0.3 # answers with scores lower than this threshold will be ignored (only works for non-rag responses)

# pdf conversion properties
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# langchain setup
vdb_prompt = PromptTemplate(
  template="""
  You are an assistant for extracting the semantic meaning of a question for querying in a vector database. Please list the semantic keywords associated with the following user query in a space separated list of disjoint words. Do not explain or say anything other than the list. Make sure to include any key words or numbers in the list. End the list with \"End\".
  
  Query: {query}

  List:
  """,
  input_variables=["query"]
)

prompt = PromptTemplate(
  template="""
  You are an assistant for question-answering tasks for information on the following Funding Opportunity Announcement (FOA) document. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Answer as a single phrase fill in the blank response. Your answer should be contained within the context. Say \"End of answer.\" when your answer is complete.

  Question: {question} 

  Definitions: {context} 

  Answer:
  """,
  input_variables=["question", "context"]
)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")
    
def trimmer(input: str):
  return re.sub(r"\s+", " ", input.strip().replace(",", " "))

# interface for pinging ai
class AI:
  print("Setting Up AI")
  client = InferenceClient(api_key=HF_API_KEY)
  embeddings = HuggingFaceEmbeddings()
  
  # chromadb vector store (inactive)
  """
  vdb_client = chromadb.PersistentClient()
  vdb = Chroma(
    embedding_function=embeddings,
    client=vdb_client,
    collection_name="vdb",
    persist_directory="./tmp/chroma_langchain_db"
  )
  """

  # FAISS vector store
  vdb = FAISS(
    embedding_function=embeddings,
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
  )

  # HuggingFace setup
  qa_llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL,
    huggingfacehub_api_token=HF_API_KEY,
    streaming=False,
    stop_sequences=["I don't know", "I don't have enough information", "End of answer"]
  )

  vdb_llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL,
    huggingfacehub_api_token=HF_API_KEY,
    streaming=False,
    stop_sequences=["End"]
  )

  retriever = vdb.as_retriever(
    search_kwargs={'k': 6}
  )

  print("Loading Documents")
  # TODO: reduce PDF to a list of definitions, make each definition a document (better for vector store)
  # currently tanks performance
  """
  loader = PyMuPDFLoader(file_path="./proposal_handbook.pdf")
  pages = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 256)
  docs = text_splitter.split_documents(pages)
  doc_ids = [f"class info doc {i}" for i in range(len(docs))]
  print(f"{len(docs)} Documents Loaded")
  print("Filling Vector Database")
  vdb.add_documents(documents=docs, ids=doc_ids)
  """
  
  vdb_chain: RunnableSequence = vdb_prompt | vdb_llm | StrOutputParser() | RunnableLambda(trimmer)
  qa_chain: RunnableSequence = prompt | qa_llm | StrOutputParser()

  # convert text into a vector
  def fe(name: str, data: str) -> numpy.ndarray:
    return AI.client.feature_extraction(text=data, prompt_name=name, model=FE_MODEL)

  # convert pdf to langchain doc splits
  def pdf_to_doc_splits(file_path: str) -> tuple[list[Document], list[Document]]:
    loader = PyMuPDFLoader(file_path=file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
    docs = text_splitter.split_documents(pages)
    return docs, pages

  # generate text
  def text_gen(prompt, max_length=500, **kwargs):
    return AI.client.text_generation(
      prompt=prompt,
      model=GEN_MODEL,
      max_new_tokens=max_length,
      stream=False,
      **kwargs
    )
  
  # try to convert a date in some format to a datetime object
  def to_date(date_str):
    date_str = date_str.strip()
    print("input date:", date_str)
    # prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant that formats a time or date into \"mm/dd/yyyy\"<|eot_id|><|start_header_id|>user<|end_header_id|>{date_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    res = AI.text_gen(f"What is \"{date_str}\" formatted as \"mm/dd/yyyy END\"? Do not explain, do not repeat prompt, do not list steps.", max_length=20, stop=["END"])
    print("response:", res)
    res = re.search(r"\d+/\d+/\d+", res)
    print(res)
    if not res: return None
    for format in ["%m/%d/%Y", "%m/%d/%y"]:
      try:
        return datetime.strptime(res.group(0), format)
      except ValueError:
        pass
    return None

  # answer a question from some context
  def qa(question, context=""):
    print(f"question: {question}")
    response = AI.client.question_answering(question=question, context=context, model=QA_MODEL)
    print(f"response: {response}")
    return response.answer if response.score >= SCORE_THRESHOLD else None
  
  # answer a question using rag (retrieval augmented generation aka: vector store)
  def rag_qa(question, foa_data, vdb_query=""):
    print(f"rag question: {question}")
    if vdb_query == "":
      vdb_query = AI.vdb_chain.invoke(input={"query": question})
    print(f"vdb query: {vdb_query}")
    context = format_docs(AI.retriever.invoke(vdb_query))
    response = AI.qa_chain.invoke({"question": question, "context": context})
    print(f"rag response: {response}")
    return response