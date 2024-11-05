from flask import Flask, send_from_directory
from dotenv import load_dotenv
from os import path, getenv

load_dotenv()

from foa import FOA
from rasr import RASR
from checklist import Checklist
from ai import AI
import pyodbc

# setup huggingface serverless api
HF_API_KEY = getenv("HF_API_KEY")
GEN_MODEL = "meta-llama/Llama-3.2-3B-Instruct" # for text generation
QA_MODEL = "deepset/roberta-base-squad2" # for question answering
SERVER_NAME = getenv("SERVER_NAME")

# setup microsoft sql server connection
"""
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      f"Server={SERVER_NAME};"
                      "Database={DB_NAME};"
                      "Trusted_Connection=yes;")
"""

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
  return AI.qa("What is my name?", context="My name is Alex.")

# test
# print(AI.to_date("8/15/24"))
# print(AI.to_date("3/31/2024"))
# print(AI.to_date("october 21 2004"))
foa = FOA(path.join(DIR, "example_foa.pdf"))
checklist = Checklist(foa, None)
