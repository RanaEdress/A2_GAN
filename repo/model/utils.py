
import re
import datetime
from typing import List, Dict

MONTH_TO_NUM = {
    "[JAN]": 1, "[FEB]": 2, "[MAR]": 3, "[APR]": 4,
    "[MAY]": 5, "[JUN]": 6, "[JUL]": 7, "[AUG]": 8,
    "[SEP]": 9, "[OCT]": 10, "[NOV]": 11, "[DEC]": 12,
}

DAY_TO_NUM = {
    "[MON]": 0, "[TUE]": 1, "[WED]": 2, "[THU]": 3,
    "[FRI]": 4, "[SAT]": 5, "[SUN]": 6,
}

REQ_WEEKDAY_SIZE = 7
REQ_MONTH_SIZE = 12
REQ_LEAP_SIZE = 2
REQ_DECADE_SIZE = 41

GRID_YEARS = 10
GRID_DAYS = 31
GRID_SIZE = GRID_YEARS * GRID_DAYS


def parse_input_line(line: str) -> List[str]:
    return line.strip().split()[:4]


def read_input_file(path: str) -> List[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [parse_input_line(line) for line in f.readlines() if line.strip()]


def condition_tokens_to_feature_ids(condition_tokens: List[str]) -> List[int]:
    day_cond, month_cond, leap_cond, decade_cond = condition_tokens

    required_weekday = DAY_TO_NUM[day_cond]
    required_month = MONTH_TO_NUM[month_cond] - 1
    required_leap = 1 if leap_cond == "[True]" else 0

    decade_prefix = int(decade_cond.replace("[", "").replace("]", ""))
    required_decade = decade_prefix - 180

    return [required_weekday, required_month, required_leap, required_decade]


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def parse_generated_date(date_str: str):
    match = re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", date_str)
    if match is None:
        return None

    try:
        d, m, y = map(int, date_str.split("-"))
        return datetime.date(y, m, d)
    except Exception:
        return None


def validate_date_against_conditions(date_str: str, conditions: List[str]) -> Dict[str, bool]:
    result = {
        "valid_format": False,
        "day_match": False,
        "month_match": False,
        "leap_match": False,
        "decade_match": False,
        "all_match": False,
    }

    dt = parse_generated_date(date_str)
    if dt is None:
        return result

    result["valid_format"] = True

    day_cond, month_cond, leap_cond, decade_cond = conditions

    result["day_match"] = dt.weekday() == DAY_TO_NUM[day_cond]
    result["month_match"] = dt.month == MONTH_TO_NUM[month_cond]

    expected_leap = True if leap_cond == "[True]" else False
    result["leap_match"] = is_leap_year(dt.year) == expected_leap

    decade_prefix = int(decade_cond.replace("[", "").replace("]", ""))
    start_year = decade_prefix * 10
    end_year = start_year + 9

    result["decade_match"] = start_year <= dt.year <= end_year

    result["all_match"] = (
        result["valid_format"]
        and result["day_match"]
        and result["month_match"]
        and result["leap_match"]
        and result["decade_match"]
    )

    return result


def index_to_year_offset_day(index: int):
    year_offset = index // GRID_DAYS
    day = (index % GRID_DAYS) + 1
    return year_offset, day


def decode_grid_prediction(logits, condition_tokens: List[str]) -> str:
    import torch

    scores = logits.detach().cpu()
    sorted_indices = torch.argsort(scores, descending=True).tolist()

    month = MONTH_TO_NUM[condition_tokens[1]]

    decade_prefix = int(condition_tokens[3].replace("[", "").replace("]", ""))
    start_year = decade_prefix * 10

    for idx in sorted_indices:
        year_offset, day = index_to_year_offset_day(idx)
        year = start_year + year_offset

        try:
            datetime.date(year, month, day)
            return f"{day}-{month}-{year}"
        except ValueError:
            continue

    return "1-1-1800"
