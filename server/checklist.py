from ai import AI
from foa import FOA
from rasr import RASR
from os import path
from datetime import datetime
from unc_calendar import UNCCalendar
import openpyxl

class Checklist:
  def __init__(self, foa: FOA, rasr: RASR):
    self.foa = foa
    self.rasr = rasr
    self.foa_data = {}
    self.data = {}

    # uncomment to output converted pdf as a file:
    # with open(path.join(path.dirname(__file__), "foa.txt"), "w", encoding="utf-8") as f:
       # f.write(self.foa.text)
    self.fetch_fields()

  def fetch_fields(self):
    # TODO: replace self.rasr with fetching specific rasr field or PI provided info
    # None fields need to be calculated later
    self.data = {
      "metadata": {
        "pi name": self.rasr,
        "grant admin unit": None, # idk where this is from
        "foa url": self.rasr,
      },
      "project overview": {
        "agency": self.foa.qa("What is the agency or organization responsible for this project?", "agency organization"),
        "code": self.foa.qa("What is the project or activity code?", "project activity code"),
        "pa/foa": self.foa.qa("What is the PA/FOA identifier aka the Funding Opporunity Number or PAR number?", "funding opportunity number identifier PAR"),
        "funding opportunity title": self.foa.qa("What is the Funding Opportunity Title?", "funding opportunity title"),
        "project duration": self.foa.qa("How long is will the project take?", "project duration years length"),
      },
      "due dates": {
        "intention to submit": None, # 4 weeks prior to sponsor deadline
        "final budget/justification": None, # 10 business days prior
        "ipf with basic science": None, # 5 business days prior
        "full proposal with final science": None, # 2 business days prior
        "final proposal to sponsor": AI.to_date(self.foa.qa("What date will this proposal be finalized? (Here are some terms you should look for: \"Full proposals must be received by\", \"Expiration date\", \"Application due date\", \"Target Date(s)\", \"Full proposal target date\", \"annually thereafter\")", "Full proposals must be received by, Expiration date, Application due date, Target Dates, Target Date, Full proposal target date,... annually thereafter")),
        "time": AI.to_date(self.foa.qa("What date was this document posted?")),
        "time zone": self.foa.qa("What is the local time zone of the applicant organization?", "time zone date")
      },
      "project details": {
        "new or renew or resub": self.rasr,
        "project period": self.rasr,
        "project title": self.rasr,
        "f&a waiver required": self.rasr,
        "associated clinical trials": self.rasr,
        "subcontracts": None,
      }
    }

    # calculate due dates
    due_dates = self.data["due dates"]
    deadline = due_dates["final proposal to sponsor"]
    if isinstance(deadline, datetime):
      due_dates["full proposal with final science"] = UNCCalendar.add_business_days(deadline, -2)
      due_dates["ipf with basic science"] = UNCCalendar.add_business_days(deadline, -5)
      due_dates["final budget/justification"] = UNCCalendar.add_business_days(deadline, -10)
      due_dates["intention to submit"] = UNCCalendar.add_business_weeks(deadline, -4)

  # WIP validators (might want to replace with actual schema library)
  # currently doesn't work
  def validate(self):
    # general validator generators
    def req_v():
      def validator(field, value):
        if value is None: return [f"required field {field} is missing"]
        if value is str and len(value) == 0: return [f"required field {field} is an empty string"]
        return []
      return validator
    
    # validates type of field
    def type_v(klass):
      return lambda field, value : [] if type(value) is None or isinstance(value, klass) else [f"{klass} field {field} is of mismatched type {type(value)}"]

    def len_v(min_length, max_length):
      return lambda field, value: [] if len(value) >= min_length and len(value) <= max_length else [f"{min_length}-{max_length} length field {field} is of length {len(value)}"]
    
    # validates list elements
    def list_v(validators):
      def validator(field, value):
        if not isinstance(value, list): return []
        errs = []
        for index, elem in enumerate(value):
          for v in validators:
            errs += validators(f"{field}[f{index}]", elem)
        return errs
      return validator
    
    # validates dict using subschema
    def dict_v(subschema):
      def validator(field, value):
        if not isinstance(subschema, dict): return []
        errs = []
        for key in subschema:
          if key not in value: value[key] = None
        for key in value:
          if key not in subschema:
            errs.append(f"field {field} contains unexpected field {key}")
            continue
          validators = subschema[key] if isinstance(subschema[key], list) else [subschema[key]]
          for v in validators:
            errs += v(f"{field}.{key}", value[key])
      return validator
    
    def enum_v(options):
      return lambda field, value: [] if value is None or value in options else [f"enum field {field} is invalid option {value}"]
      
    # schema
    # field => [validators]
    schema = [req_v(), type_v(dict), dict_v({
      "metadata": [type_v(dict), {
        "pi name": [type_v(str), req_v()],
        "grand admin unit": [type_v(str), req_v()]
      }],
      "project overview": [req_v(), type_v(dict), dict_v({
        "agency": type_v(str),
        "code": type_v(str),
        "pa/foa": type_v(str),
        "funding opportunity title": type_v(str),
        "project duration": type_v(str)
      })],
      "due dates": [req_v(), type_v(dict), dict_v({
        "intention to submit": type_v(datetime),
        "final budget justification": type_v(datetime),
        "ipf with basic science": type_v(datetime),
        "full proposal with final science": type_v(datetime),
        "final proposal to sponsor": type_v(datetime),
        "time": type_v(datetime),
        "time zone": type_v(str)
      })],
      "project details": [req_v(), type_v(dict), dict_v({
        "new or renew or resub": [req_v(), type_v(str)],
        "project period": [req_v(), type_v(list), len_v(2), list_v([req_v(), type_v(datetime)])],
        "project title": [req_v(), type_v(str)],
        "f&a waiver required": [req_v(), enum_v(["yes", "no", "unknown"])],
        "associated clinical trials": [req_v(), enum_v(["yes", "no", "unknown"])],
        "subcontracts": [req_v(), type_v(list), list_v([req_v(), type_v(dict), dict_v({
          "name": [req_v(), type_v(str)],
          "location": [req_v(), type_v(str)],
          "work conducted at location": [req_v(), enum_v(["yes", "no"])]
        })])],
        "consultants": [req_v(), type_v(list), list_v([req_v(), type_v(dict), dict_v({
          "name": [req_v(), type_v(str)],
          "email": [req_v(), type_v(str)],
          "coi questions": [req_v(), type_v(dict), dict_v({
            "substantially contributed to design of study": [req_v(), enum_v(["yes", "no"])],
            "conducting experiments or activities": [req_v(), enum_v(["yes", "no"])],
            "directly involved or have control over collection of data": [req_v(), enum_v(["yes", "no"])],
            "involved in analysis of data": [req_v(), enum_v(["yes", "no"])],
            "author on public dissemination": [req_v(), enum_v(["yes", "no"])],
          })],
          "conflict of interest": [req_v(), enum_v(["yes", "no"])]
        })])]
      })],
      "proposal prep details": [req_v(), type_v(dict), dict_v({
        "format or font requirements": type_v(str)
      })],
      "administrative components": [req_v(), type_v(dict), dict_v({
        # TODO
      })]
    })]

    # validate schema
    errs = []
    for validator in schema:
      errs += validator("", self.data)
    return errs

  def fill_checklist(self):
    workbook = openpyxl.load_workbook("../ProposalChecklist.xlsx")
    sheet = workbook["Checklist Template"]

    sheet["B5"] = self.data["project overview"]["agency"]
    sheet["B6"] = self.foa_data["project overview"]["code"]
    sheet["B7"] = self.foa_data["project overview"]["pa/foa"]
    sheet["B8"] = self.foa_data["project overview"]["funding opportunity title"]
    sheet["B9"] = self.foa_data["project overview"]["project duration"]

    sheet["B13"] = self.data["due dates"]["intention to submit"]
    sheet["B14"] = self.data["due dates"]["final budget justification"]
    sheet["B15"] = self.data["due dates"]["ipf with basic science"]
    sheet["B16"] = self.data["due dates"]["full proposal with final science"]
    sheet["B17"] = self.data["due dates"]["final proposal to sponsor"]
    sheet["B18"] = self.data["due dates"]["time"] + " " + self.data["due dates"]["time zone"]


    workbook.save("../checklist_modified.xlsx")