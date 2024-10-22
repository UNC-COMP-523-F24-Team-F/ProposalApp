from ai import AI
from foa import FOA
from rasr import RASR
from os import path

class Checklist:
  def __init__(self, foa: FOA, rasr: RASR):
    self.foa = foa
    self.rasr = rasr

    # uncomment to output converted pdf as a file:
    # with open(path.join(path.dirname(__file__), "foa.txt"), "w", encoding="utf-8") as f:
       # f.write(self.foa.text)
    self.fetch_foa_fields()

  def fetch_foa_fields(self):
    self.foa_data = {
      "agency": self.foa.qa("What is the agency or organization responsible for this project?"),
      "code": self.foa.qa("What is the project or activity code?"),
      "funding_opportunity_title": self.foa.qa("What is the funding opportunity title?"),
      "project_duration": self.foa.qa("What is the project duration?"),
      "time": AI.to_date(self.foa.qa("What date was this document posted?")),
    }

    print(self.foa_data)


