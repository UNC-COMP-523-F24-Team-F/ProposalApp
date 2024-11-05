from os import path
from ai import *

class FOA:
  def __init__(self, file_path):
    print("LOADING FOA")
    docs: list[Document] = AI.pdf_to_doc_splits(file_path)
    self.doc_ids = [f"FOA {i}" for i in range(len(docs))]
    print("WRITING FOA")
    AI.vdb.add_documents(documents=docs, ids=self.doc_ids)
    print("FOA LOADED")

  def __del__(self):
    print("DELETING FOA")
    AI.vdb.delete(self.doc_ids)
    print("FOA DELETED")

  def qa(self, prompt):
    return AI.rag_qa(prompt)
