from flask import Flask, send_from_directory
from dotenv import load_dotenv
from os import path, getenv
from huggingface_hub import InferenceClient

load_dotenv()

# setup huggingface serverless api
HF_API_KEY = getenv("HF_API_KEY")
GEN_MODEL = "meta-llama/Llama-3.2-3B-Instruct" # for text generation
QA_MODEL = "deepset/roberta-base-squad2" # for question answering

class AI:
  client = InferenceClient(api_key=HF_API_KEY)

  def text_gen(prompt):
    return AI.client.text_generation(
      prompt=prompt,
      model=GEN_MODEL,
      max_new_tokens=500,
      stream=False,
    )

  def qa(question, context=""):
    response = AI.client.question_answering(question=question, context=context, model=QA_MODEL)
    return response.answer

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
  return AI.qa("What is my name?", context="My name is Joe Biden.")
