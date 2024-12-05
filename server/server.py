from flask import Flask, send_from_directory
from dotenv import load_dotenv
from os import path, getenv
import json

load_dotenv()

from foa import FOA
from rasr import RASR
from checklist import Checklist
from ai import AI

app = Flask(__name__)

DIR = path.dirname(__file__)
BUILD_PATH = path.join(DIR, '..', 'build', 'browser')

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
  return send_from_directory(BUILD_PATH, path)

@app.route("/")
def index():
  print(BUILD_PATH)
  return send_from_directory(BUILD_PATH, 'index.html')

# ping huggingface ai
@app.route('/ping_ai', methods=['GET'])
def ping_ai():
  print("PINGING QA")
  AI.qa("What is my name?", context="My name is Alex.")
  print("PINGING RAG QA")
  AI.rag_qa("test", "", "")

# DRIVER CODE
# print(AI.to_date("8/15/24"))
# print(AI.to_date("3/31/2024"))
# print(AI.to_date("october 21 2004"))
ping_ai() # ensure hf is working (goes out of date often)
foa = FOA(path.join(DIR, "example_foa.pdf"))
checklist = Checklist(foa, None)
checklist.fill_checklist(path.join(DIR, "..", "ChecklistTemplate.xlsx"), path.join(DIR, "OutputChecklist.xlsx"))

print(json.dumps(checklist.data, indent=4, sort_keys=True, default=str))