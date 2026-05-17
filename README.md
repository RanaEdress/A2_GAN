# Assignment 2 - Date Generation

## Problem Formulation

The task is to generate a date that satisfies four input conditions: weekday, month, leap-year status, and decade.

A key challenge is that each input can have multiple correct dates. Therefore, I used a multi-valid **grid-based generation** formulation instead of forcing the model to reproduce one fixed target date.

This is not a candidate classification setup. During inference, the model receives only the input conditions and generates a date by predicting a score distribution over a fixed grid of possible day/year positions. No candidate date is provided as input.

For each input condition, the model outputs 310 scores:

10 years in the requested decade × 31 possible days = 310 possible outputs.

The month and decade are fixed by the input, and the model learns to select a valid day and year offset. The highest-scoring valid grid position is decoded into the final generated date.

## Project Structure

```text
data/
  data.txt
  example_input.txt

model/
  predict.py
  models.py
  utils.py
  evaluate.py
  train.py
  weights/
    ae_grid_generator.pt
    gan_grid_generator.pt
    cvae_grid_generator.pt
    mlp_grid_generator.pt

environment.yml
README.md
```

## Models

Four models were implemented using the same grid-based generation formulation:

| Model | Category |
|---|---|
| AE Grid Generator | Course model |
| GAN Grid Generator | Course model / required GAN |
| CVAE Grid Generator | Outside-course extension |
| MLP Grid Generator | Outside-course model |

## Results

| Model | All Match |
|---|---:|
| MLP Grid Generator | 1.0000 |
| CVAE Grid Generator | 0.9800 |
| AE Grid Generator | 0.9767 |
| GAN Grid Generator | 0.6933 |

The MLP achieved the best result because this task is a structured generation problem from four input conditions to a fixed grid of possible dates. AE and CVAE also performed strongly because the grid formulation allowed them to model multiple valid outputs. GAN performed lower, which is expected because adversarial training is less stable for structured symbolic generation.

## Inference

The inference script is located at:

```text
model/predict.py
```

From inside the `model` folder, run:

```bash
python predict.py -i ../data/example_input.txt -o ../predictions.txt
```

The output format matches `data.txt` exactly:

```text
[DAY] [MONTH] [LEAP] [DECADE] generated-date
```

Example output:

```text
[WED] [JAN] [False] [180] 28-1-1801
```

## Environment

Create the conda environment using:

```bash
conda env create -f environment.yml
conda activate dates-generator
```

## Final Inference Result

On the provided `example_input.txt`, the final MLP model achieved:

| Metric | Score |
|---|---:|
| valid_format | 1.0000 |
| day_match | 0.9966 |
| month_match | 1.0000 |
| leap_match | 1.0000 |
| decade_match | 1.0000 |
| all_match | 0.9966 |

## Conclusion

The grid-based formulation solved the multiple-correct-answer problem by allowing the model to assign high scores to all valid dates. The final selected model is the MLP Grid Generator because it achieved the highest all-condition match score. The final system performs structured date generation from conditions only, without receiving candidate dates during inference.
