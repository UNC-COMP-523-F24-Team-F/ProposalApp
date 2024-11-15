from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar, AbstractHolidayCalendar, Holiday
from pandas import Timestamp

# class to handle business day calculations
class UNCCalendar:
  class UNCHolidayCalendar(AbstractHolidayCalendar):
    # TODO: get list of holidays from external database specific to UNC
    # for now uses US federal holidays
    rules = USFederalHolidayCalendar.rules

  calendar = UNCHolidayCalendar()
  business_day = CustomBusinessDay(calendar=calendar)

  # returns whether current date is a holiday
  def is_holiday(date: datetime) -> bool:
    workdate: Timestamp = date + UNCCalendar.business_day * 0
    return workdate.to_pydatetime() != date

  # returns the date k business days after the given date
  def add_business_days(date: datetime, k: int) -> datetime:
    date: Timestamp = date + UNCCalendar.business_day * k
    return date.to_pydatetime()
  
  # returns the date k business weeks after the given date
  def add_business_weeks(date: datetime, k: int) -> datetime:
    # offset by k weeks
    date += timedelta(days = k * 7)

    # deal with landing on holidays
    if UNCCalendar.is_holiday(date):
      date: Timestamp = date + UNCCalendar.business_day if k >= 0 else date - UNCCalendar.business_day
      date = date.to_pydatetime()
    
    return date