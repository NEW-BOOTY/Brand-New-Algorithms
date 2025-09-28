# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import json  # For parsing
import logging
logging.basicConfig(level=logging.INFO)

def generate_modular_scripts(infra_state, user_intent):
    tasks = json.loads(user_intent)['tasks']  # Detailed parse
    resources = infra_state['resources']
    scripts = []
    for task in tasks:
        script = f"#!/bin/bash\necho 'Running {task} on {resources[0]}'\n"  # Generate bash
        scripts.append(script)
    return scripts