from flask import Flask, send_from_directory
from dotenv import load_dotenv
from os import path, getenv

load_dotenv()

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
