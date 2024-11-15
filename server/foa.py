from os import path
from ai import *

class FOA:
  def __init__(self, file_path):
    print("LOADING FOA")
    splits, pages = AI.pdf_to_doc_splits(file_path)
    self.data = "\n\n".join([page.page_content for page in pages])
    self.doc_ids = [f"FOA {i}" for i in range(len(splits))]
    print("WRITING FOA")
    AI.vdb.add_documents(documents=splits, ids=self.doc_ids)
    print("FOA LOADED")

  def __del__(self):
    print("DELETING FOA")
    AI.vdb.delete(self.doc_ids)
    print("FOA DELETED")

  def qa(self, prompt, vdb_query=""):
    return AI.rag_qa(prompt, self.data, vdb_query=vdb_query)
