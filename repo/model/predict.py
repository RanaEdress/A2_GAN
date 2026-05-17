
import argparse
import os
import torch

from utils import (
    read_input_file,
    condition_tokens_to_feature_ids,
    decode_grid_prediction,
)

from models import build_model


DEFAULT_MODEL = "mlp"

WEIGHT_FILES = {
    "ae": "ae_grid_generator.pt",
    "gan": "gan_grid_generator.pt",
    "cvae": "cvae_grid_generator.pt",
    "mlp": "mlp_grid_generator.pt",
}


def load_model(model_name: str, device):
    model = build_model(model_name)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "weights", WEIGHT_FILES[model_name])

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict_file(input_path: str, output_path: str, model_name: str = DEFAULT_MODEL):
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name, device)

    condition_lines = read_input_file(input_path)

    outputs = []

    with torch.no_grad():
        for cond_tokens in condition_lines:
            features = torch.tensor(
                [condition_tokens_to_feature_ids(cond_tokens)],
                dtype=torch.long,
                device=device
            )

            logits = model(features)[0]
            pred_date = decode_grid_prediction(logits, cond_tokens)

            output_line = " ".join(cond_tokens) + " " + pred_date
            outputs.append(output_line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line + "\n")

    print(f"Saved predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["ae", "gan", "cvae", "mlp"]
    )

    args = parser.parse_args()

    predict_file(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model
    )


if __name__ == "__main__":
    main()
