import time
from datetime import datetime, timedelta

def add_hours_to_date(hours, date=None):
    if date is None:
        date = datetime.now()

    next_date = date + timedelta(hours=hours)
    return next_date

def get_formatted_date(date):
    return date.strftime('%H:%M %d/%m/%Y')