from os import getenv
from huggingface_hub import InferenceClient
from datetime import datetime
import re

# setup huggingface serverless api
HF_API_KEY = getenv("HF_API_KEY")
GEN_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct" # for text generation
QA_MODEL = "deepset/roberta-base-squad2" # for question answering
SERVER_NAME = getenv("SERVER_NAME")
SCORE_THRESHOLD = 0.3 # answers with scores lower than this threshold will be ignored

# interface for pinging ai
class AI:
  client = InferenceClient(api_key=HF_API_KEY)

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
  