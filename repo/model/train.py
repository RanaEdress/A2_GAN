
"""
Training code summary.

The submitted notebook contains the full training pipeline for:
1. AE Grid Generator
2. GAN Grid Generator
3. CVAE Grid Generator
4. MLP Grid Generator

All models use the same formulation:

conditions -> 310 grid logits

where:
310 = 10 years in the required decade × 31 possible days.

Trained weights are stored under:
model/weights/

Inference:
python predict.py -i ../data/example_input.txt -o ../predictions.txt
"""

if __name__ == "__main__":
    print("Full training is provided in the submitted notebook.")
    print("Use predict.py for inference.")
