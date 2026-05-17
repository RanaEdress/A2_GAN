
import argparse
from utils import validate_date_against_conditions


def evaluate_predictions(prediction_file: str):
    lines = []

    with open(prediction_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line.strip())

    total = 0
    valid_format = 0
    day_match = 0
    month_match = 0
    leap_match = 0
    decade_match = 0
    all_match = 0

    for line in lines:
        parts = line.split()

        cond_tokens = parts[:4]
        pred_date = parts[4]

        result = validate_date_against_conditions(pred_date, cond_tokens)

        total += 1
        valid_format += int(result["valid_format"])
        day_match += int(result["day_match"])
        month_match += int(result["month_match"])
        leap_match += int(result["leap_match"])
        decade_match += int(result["decade_match"])
        all_match += int(result["all_match"])

    metrics = {
        "valid_format": valid_format / total,
        "day_match": day_match / total,
        "month_match": month_match / total,
        "leap_match": leap_match / total,
        "decade_match": decade_match / total,
        "all_match": all_match / total,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", required=True)
    args = parser.parse_args()

    evaluate_predictions(args.predictions)


if __name__ == "__main__":
    main()
