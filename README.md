# Assignment 2 - Date Generation

## Problem Formulation

The task is to generate a date that satisfies four input conditions: weekday, month, leap-year status, and decade.

A key challenge is that each input can have multiple correct dates. Therefore, I used a multi-valid grid generation formulation instead of forcing the model to reproduce one fixed target date.

For each input condition, the model outputs 310 scores:

10 years in the requested decade × 31 possible days = 310 possible outputs.

The month and decade are fixed by the input, and the model learns to select a valid day and year offset.

## Models

Four models were implemented using the same grid formulation:

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

The MLP achieved the best result because this task is a structured mapping problem from four input conditions to a fixed grid of possible dates. AE and CVAE also performed strongly because the grid formulation allowed them to model multiple valid outputs. GAN performed lower, which is expected because adversarial training is less stable for structured symbolic generation.

## Inference

The inference script is located at:

`model/predict.py`

It can be run using:

`python predict.py -i ../data/example_input.txt -o ../predictions.txt`

The output format is:

`conditions + generated_date`

## Conclusion

The grid-based formulation solved the multiple-correct-answer problem by allowing the model to assign high scores to all valid dates. The final selected model is the MLP Grid Generator because it achieved the highest all-condition match score.
