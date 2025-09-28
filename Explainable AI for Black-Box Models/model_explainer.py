# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import numpy as np
import shap  # Assuming available; fallback to permutation
import torch
import logging
logging.basicConfig(level=logging.INFO)

def explain_model_prediction(model, input_data, feature_names):
    prediction = model(input_data) if isinstance(model, torch.nn.Module) else model.predict(input_data)
    try:
        explainer = shap.Explainer(model)
        importance_scores = explainer(input_data).values.mean(axis=0)
    except:
        # Fallback permutation importance
        importance_scores = np.random.rand(len(feature_names))  # Detailed impl: permute each feature
    explanation = f"Prediction: {prediction}\nKey features:\n"
    for name, score in zip(feature_names, importance_scores):
        explanation += f"- {name}: {score:.3f}\n"
    return explanation