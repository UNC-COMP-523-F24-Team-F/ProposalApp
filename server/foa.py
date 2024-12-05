from os import path
from ai import *

# represents an FOA document
# loads into vector store on construction
# removes from vector store on destruction
# ensure only one exists at a time to prevent interference.
class FOA:
  id_offset = 0 # can't re-add docs with deleted ids
  def __init__(self, file_path):
    print("LOADING FOA")
    splits, pages = AI.pdf_to_doc_splits(file_path)
    self.data = "\n\n".join([page.page_content for page in pages])
    self.doc_ids = [f"FOA {i+FOA.id_offset}" for i in range(len(splits))]
    FOA.id_offset += len(self.doc_ids)
    print("WRITING FOA")
    AI.vdb.add_documents(documents=splits, ids=self.doc_ids)
    print("FOA LOADED")

  def __del__(self):
    print("DELETING FOA")
    AI.vdb.delete(self.doc_ids)
    print("FOA DELETED")

  def qa(self, prompt, vdb_query=""):
    return AI.rag_qa(prompt, self.data, vdb_query=vdb_query)
