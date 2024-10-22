from pypdf import PdfReader
from os import path
from ai import AI

class FOA:
  def __init__(self, file_path):
    self.reader = PdfReader(file_path)

    # foa document
    pages = [page.extract_text() for page in self.reader.pages]
    self.text = "\n\n*page break*\n\n".join(pages)
  
  def qa(self, prompt):
    return AI.qa(prompt, self.text)
