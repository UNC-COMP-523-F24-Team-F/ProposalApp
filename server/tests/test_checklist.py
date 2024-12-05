import openpyxl
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.checklist import Checklist


def assertFillChecklist():
    checklist = Checklist()
    path = "./test.xlsx"
    checklist.fetch_fields()
    checklist.fill_checklist(path)

    workbook = openpyxl.load_workbook("./checklist_modified.xlsx")
    sheet = workbook["Checklist Template"]

    assert sheet["B5"] == checklist.data["project overview"]["agency"]
    assert sheet["B6"] == checklist.foa_data["project overview"]["code"]
    assert sheet["B7"] == checklist.foa_data["project overview"]["pa/foa"]
    assert sheet["B8"] == checklist.foa_data["project overview"]["funding opportunity title"]
    assert sheet["B9"] == checklist.foa_data["project overview"]["project duration"]

    assert sheet["B13"] == checklist.data["due dates"]["intention to submit"]
    assert sheet["B14"] == checklist.data["due dates"]["final budget justification"]
    assert sheet["B15"] == checklist.data["due dates"]["ipf with basic science"]
    assert sheet["B16"] == checklist.data["due dates"]["full proposal with final science"]
    assert sheet["B17"] == checklist.data["due dates"]["final proposal to sponsor"]
    assert sheet["B18"] == checklist.data["due dates"]["time"] + " " + checklist.data["due dates"]["time zone"]
    
    