# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import torch
import matplotlib.pyplot as plt
import io
import base64
import logging
logging.basicConfig(level=logging.INFO)

def visual_explainer(model, input_data):
    activation_maps = []
    hooks = []
    def hook_fn(module, input, output):
        activation_maps.append(output.detach().cpu().numpy())
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))
    model(input_data)
    for hook in hooks:
        hook.remove()
    
    # Generate base64 image
    fig, ax = plt.subplots()
    ax.imshow(activation_maps[0][0])  # Example heatmap
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"image": img_base64}