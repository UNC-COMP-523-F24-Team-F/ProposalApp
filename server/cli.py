from server import ping_ai
from foa import FOA
from rasr import RASR
from checklist import Checklist
from os import path

ping_ai() # ensure hf is working

import json

DIR = path.dirname(__file__)

# CLI FOA to Checklist
# takes in FOA path, generates checklist file (output stored in OutputChecklist.xlsx)

while True:
  # get path of foa
  foa_path = input("Enter foa path relative to ./server (leave blank to exit): ")
  if len(foa_path) == 0:
    break
  foa_path = path.join(DIR, foa_path)
  if not path.exists(foa_path):
    print("file not found")
    continue

  # generate checklist
  foa = FOA(foa_path)
  checklist = Checklist(foa, None)
  print(json.dumps(checklist.data, indent=4, sort_keys=True, default=str))

  # save checklist as excel file
  checklist.fill_checklist(path.join(DIR, "ChecklistTemplate.xlsx"), path.join(DIR, "OutputChecklist.xlsx"))

print("Closed Gracefully")