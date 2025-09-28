# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import numpy as np
import torch
import tensorflow as tf
import scipy.stats as stats  # For Gaussian noise optimization
import logging
logging.basicConfig(level=logging.INFO)

def differential_privacy_engine(model, data, epsilon, delta, clip_norm, framework='pytorch'):
    for batch in data:
        gradients = compute_gradients(model, batch, framework)
        
        # Vectorized clipping
        clipped_gradients = clip_gradients(gradients, clip_norm)
        
        # Optimized noise with SciPy
        noisy_gradients = add_gaussian_noise(clipped_gradients, epsilon, delta)
        
        apply_gradients(model, noisy_gradients, framework)
        logging.info(f"Processed batch with noise sigma={np.sqrt(2 * np.log(1.25 / delta)) / epsilon:.4f}")
    
    return model

def compute_gradients(model, batch, framework):
    data, target = batch
    if framework == 'pytorch':
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        return [p.grad.detach().cpu().numpy() for p in model.parameters()]
    else:
        with tf.GradientTape() as tape:
            output = model(data)
            loss = tf.keras.losses.CategoricalCrossentropy()(target, output)
        return tape.gradient(loss, model.trainable_variables)

def clip_gradients(gradients, clip_norm):
    total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients))
    scale = min(1.0, clip_norm / max(1e-6, total_norm))
    return [g * scale for g in gradients]

def add_gaussian_noise(gradients, epsilon, delta):
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = [stats.norm.rvs(loc=0, scale=sigma, size=g.shape) for g in gradients]
    return [g + n for g, n in zip(gradients, noise)]

def apply_gradients(model, gradients, framework):
    if framework == 'pytorch':
        for p, g in zip(model.parameters(), gradients):
            p.data -= g  # Simple update; add optimizer for prod
    else:
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))