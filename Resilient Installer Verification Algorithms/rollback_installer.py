# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import os
import shutil  # For file ops
import logging
logging.basicConfig(level=logging.INFO)

def rollback_aware_installer(install_steps, system_state):
    change_log = []
    try:
        for step in install_steps:
            change = execute_step(step, system_state)
            change_log.append(change)
            if not validate_step(step, system_state):
                raise ValueError(f"Validation failed for {step['id']}")
        return True, change_log
    except Exception as e:
        rollback_changes(change_log)
        return False, str(e)

def execute_step(step, system_state):
    if step['type'] == 'file_copy':
        dest = os.path.join(system_state['dir'], step['file'])
        shutil.copy(step['source'], dest)
        return {'type': 'file_copy', 'dest': dest, 'source': step['source']}
    # Add more step types...

def validate_step(step, system_state):
    if step['type'] == 'file_copy':
        return os.path.exists(os.path.join(system_state['dir'], step['file']))
    return True

def rollback_changes(change_log):
    for change in reversed(change_log):
        if change['type'] == 'file_copy':
            os.remove(change['dest'])
            logging.info(f"Rolled back {change['dest']}")