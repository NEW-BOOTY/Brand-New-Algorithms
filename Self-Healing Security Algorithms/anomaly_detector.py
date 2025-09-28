# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import numpy as np
import scipy.stats as stats
import collections  # For deque window
import logging
logging.basicConfig(level=logging.INFO)

def anomaly_detector(events, window_size, threshold):
    behavior_model = {"mean": 0, "std": 1}
    event_window = collections.deque(maxlen=window_size)
    
    for event in events:
        event_window.append(event['value'])
        update_behavior_model(behavior_model, event_window)
        anomaly_score = stats.zscore([event['value']], behavior_model['mean'], behavior_model['std'])[0]
        if abs(anomaly_score) > threshold:
            logging.warning(f"ALERT: Anomaly in event {event['id']}, score: {anomaly_score:.4f}")
    
    return behavior_model

def update_behavior_model(model, event_window):
    arr = np.array(event_window)
    model['mean'] = np.mean(arr)
    model['std'] = np.std(arr) if len(arr) > 1 else 1.0