# Tech Talk demo, also useful MVP for the full stack

from os import getenv
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
import faiss

load_dotenv()

# Huggingface Serverless API LLMs Setup
print("Setting Up Huggingface Serverless API")
HF_API_KEY = getenv("HF_API_KEY")
GEN_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
client = InferenceClient(api_key=HF_API_KEY)
embeddings = HuggingFaceEmbeddings()

# llm for question answering
qa_llm = HuggingFaceEndpoint(
  repo_id=GEN_MODEL,
  huggingfacehub_api_token=HF_API_KEY,
  streaming=False,
  stop_sequences=["I don't know", "I don't have enough information", "End of answer"]
)

# llm for vector database query generation
vdb_llm = HuggingFaceEndpoint(
  repo_id=GEN_MODEL,
  huggingfacehub_api_token=HF_API_KEY,
  streaming=False,
  stop_sequences=["End"]
)

# FAISS Vector Database Setup
# create database
print("Creating FAISS Vector Database")
vdb = FAISS(
  embedding_function=embeddings,
  index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
  docstore=InMemoryDocstore(),
  index_to_docstore_id={}
)
vdb_retriever = vdb.as_retriever(
  search_kwargs={'k': 6}
)

# load pdfs into the database
print("Loading Documents")
loader = PyMuPDFLoader(file_path="./class_info.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 256)
docs, _ = text_splitter.split_documents(pages)
doc_ids = [f"class info doc {i}" for i in range(len(docs))]
print(f"{len(docs)} Documents Loaded")
print("Filling Vector Database")
vdb.add_documents(documents=docs, ids=doc_ids)

# LangChain Chain Setup
print("Creating Chains")
# prompts
demo_prompt = PromptTemplate(
  template="""
  You are an assistant for question-answering tasks about the courses for UNC's Computer Science department. Use the following pieces of retrieved context taken from a PDF to answer the question. If you don't know the answer, just say that you don't know. Answer consisely in three sentences or less. Say \"End of answer.\" when your answer is complete.

  Question: {question} 

  Context: {context} 

  Answer:
  """,
  input_variables=["question", "context"]
)

vdb_prompt = PromptTemplate(
  template="""
  You are an assistant for extracting the semantic meaning of a question for querying in a vector database. Please list the semantic keywords associated with the following user query in a space separated list of disjoint words. Do not explain or say anything other than the list. If a course is specified, include it in the list. End the list with \"End\".
  
  Query: {query}

  List:
  """,
  input_variables=["query"]
)

# Chains
# output format helper
def trimmer(input: str):
  return re.sub(r"\s+", " ", input.strip().replace(",", " "))

# vector database query generation chain
vdb_chain: RunnableSequence = vdb_prompt | vdb_llm | StrOutputParser() | RunnableLambda(trimmer)

# question answering chain
qa_chain: RunnableSequence = demo_prompt | qa_llm | StrOutputParser()

# prompt
while True:
  question = input("\nQuestion: ")
  if len(question) == 0:
    break

  # generate vector database query
  vdb_query = vdb_chain.invoke(input={"query": question})
  print(f"\nVector Database Query: {vdb_query}")

  # get context from vector database
  context_docs = vdb_retriever.invoke(vdb_query)
  context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}\n\nMetadata: {doc.metadata}" for i, doc in enumerate(context_docs)])

  # perform question answering with retrieved context
  response = qa_chain.invoke(input={"question": question, "context": context_docs})
  print(f"\nResponse: {response}")

  # display context
  if input(f"\nShow Context? (y/N): ").strip().lower() == "y":
    print(f"\nContext: {context}")