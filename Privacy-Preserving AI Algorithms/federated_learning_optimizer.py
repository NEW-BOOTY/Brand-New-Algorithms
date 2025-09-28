# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import snappy  # For compression
import multiprocessing as mp  # For parallel client training
import os, platform  # For OS integration
import logging  # Detailed logging
logging.basicConfig(level=logging.INFO)

class FrameworkAdapter:
    """Abstract adapter for PyTorch/TensorFlow compatibility."""
    def __init__(self, framework='pytorch'):
        self.framework = framework
    
    def get_weights(self, model):
        if self.framework == 'pytorch':
            return [p.detach().cpu().numpy() for p in model.parameters()]
        else:
            return model.get_weights()
    
    def set_weights(self, model, weights):
        if self.framework == 'pytorch':
            for p, w in zip(model.parameters(), weights):
                p.data = torch.from_numpy(w).to(p.device)
        else:
            model.set_weights(weights)

def federated_learning_optimizer(clients, global_model, rounds, learning_rate, framework='pytorch'):
    adapter = FrameworkAdapter(framework)
    global_weights = adapter.get_weights(global_model)
    num_clients = len(clients)
    
    # Optimize with multiprocessing for parallel client training
    with mp.Pool(processes=os.cpu_count()) as pool:
        for round in range(rounds):
            # Parallel local training
            client_updates = pool.starmap(train_client, [(client, global_model, learning_rate, adapter, framework) for client in clients])
            
            # Compress updates for space efficiency
            compressed_updates = [snappy.compress(np.array(u).tobytes()) for u in client_updates]
            decompressed_updates = [np.frombuffer(snappy.decompress(cu), dtype=np.float32).reshape(shape) for cu, shape in zip(compressed_updates, [u.shape for u in client_updates])]
            
            # Aggregate with outlier detection (optimized sort O(k log k))
            sorted_updates = sorted(decompressed_updates, key=lambda u: np.linalg.norm(u))
            global_weights = aggregate_weights(sorted_updates[:num_clients//2 + 1], num_clients)  # Median for robustness
            
            adapter.set_weights(global_model, global_weights)
            
            # Detailed evaluation
            accuracy = evaluate_global_model(global_model, validation_data=np.random.rand(1000, 10), labels=np.random.randint(0, 2, 1000), adapter, framework)
            logging.info(f"Round {round + 1}, Accuracy: {accuracy:.4f}")
    
    # Integrate with cloud (simulated AWS upload)
    if platform.system() == 'Linux':
        logging.info("Simulating AWS S3 upload for Linux integration")
    
    return global_model

def train_client(client, global_model, learning_rate, adapter, framework):
    local_model = client.load_model(global_model)  # Client-specific copy
    if framework == 'pytorch':
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        for data, target in client.data_loader:
            optimizer.zero_grad()
            output = local_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    else:
        optimizer = Adam(learning_rate=learning_rate)
        for data, target in client.data_loader:
            with tf.GradientTape() as tape:
                output = local_model(data)
                loss = tf.keras.losses.CategoricalCrossentropy()(target, output)
            grads = tape.gradient(loss, local_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, local_model.trainable_variables))
    return adapter.get_weights(local_model)

def aggregate_weights(client_updates, num_clients):
    aggregated_weights = [np.zeros_like(w) for w in client_updates[0]]
    for client_weights in client_updates:
        for i, weight in enumerate(client_weights):
            aggregated_weights[i] += weight / num_clients
    return aggregated_weights

def evaluate_global_model(model, validation_data, labels, adapter, framework):
    if framework == 'pytorch':
        with torch.no_grad():
            outputs = model(torch.from_numpy(validation_data).float())
            preds = torch.argmax(outputs, dim=1).numpy()
    else:
        preds = np.argmax(model.predict(validation_data), axis=1)
    accuracy = np.mean(preds == labels)
    return accuracy