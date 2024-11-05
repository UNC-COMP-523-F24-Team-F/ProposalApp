from os import getenv
from huggingface_hub import InferenceClient
from datetime import datetime
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.messages.base import BaseMessage
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import chromadb
import re
import numpy

# setup huggingface serverless api
HF_API_KEY = getenv("HF_API_KEY")
GEN_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct" # for text generation
FE_MODEL = "thenlper/gte-large" # for feature extraction
QA_MODEL = "deepset/roberta-base-squad2" # for question answering
SERVER_NAME = getenv("SERVER_NAME")
SCORE_THRESHOLD = 0.3 # answers with scores lower than this threshold will be ignored

# pdf conversion properties
CHUNK_SIZE = 512
CHUNK_OVERLAP = 96

# langchain setup
prompt = PromptTemplate(
  template="""
  You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Answer as a single phrase fill in the blank response. Your answer should be contained within the context. Say \"End of answer.\" when your answer is complete.

  Question: {question} 

  Context: {context} 

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

# interface for pinging ai
class AI:
  client = InferenceClient(api_key=HF_API_KEY)
  vdb_client = chromadb.PersistentClient()
  embeddings = HuggingFaceEmbeddings()
  
  vdb = Chroma(
    embedding_function=embeddings,
    client=vdb_client,
    collection_name="vdb",
    persist_directory="./tmp/chroma_langchain_db"
  )

  qa_llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL,
    huggingfacehub_api_token=HF_API_KEY,
    streaming=False,
    stop_sequences=["I don't know", "I don't have enough information", "End of answer"]
  )

  retriever = vdb.as_retriever(k=6)
  
  qa_chain: RunnableSequence = prompt | qa_llm | StrOutputParser()

  # convert text into a vector
  def fe(name: str, data: str) -> numpy.ndarray:
    return AI.client.feature_extraction(text=data, prompt_name=name, model=FE_MODEL)

  # convert pdf to langchain doc splits
  def pdf_to_doc_splits(file_path: str, detailed=False) -> list[Document]:
    if detailed:
      # currently doesn't work on windows (libmagic doesn't exist)
      loader = UnstructuredLoader(
        file_path=file_path,
        strategy="fast",
        partition_via_api=False,
        coordinates=False
      )
      docs = loader.load()
    else:
      loader = PyMuPDFLoader(file_path=file_path)
      pages = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
      docs = text_splitter.split_documents(pages)
    return docs

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
  
  # answer a question using rag
  def rag_qa(question):
    print(f"rag question: {question}")
    context = format_docs(AI.retriever.invoke(question))
    response = AI.qa_chain.invoke({"question": question, "context": context})
    print(f"rag response: {response}")
    return response