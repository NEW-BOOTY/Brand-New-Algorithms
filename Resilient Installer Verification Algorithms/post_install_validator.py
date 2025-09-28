# Copyright © 2025 Devin B. Royal. All Rights Reserved.

import os
import hashlib
import subprocess
import logging
logging.basicConfig(level=logging.INFO)

def post_install_validator(install_dir, expected_files, expected_permissions, services):
    results = []
    for file in expected_files:
        path = os.path.join(install_dir, file['name'])
        if not verify_file_integrity(path, file['checksum']):
            results.append(f"Integrity failed: {path}")
    for file, perms in expected_permissions.items():
        path = os.path.join(install_dir, file)
        if not verify_permissions(path, perms):
            results.append(f"Permissions failed: {path}")
    for service in services:
        if not check_service_running(service):
            results.append(f"Service not running: {service}")
    return results

def verify_file_integrity(file_path, expected_checksum):
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest() == expected_checksum
    except Exception as e:
        logging.error(f"Integrity error: {e}")
        return False

def verify_permissions(file_path, expected_perms):
    return oct(os.stat(file_path).st_mode)[-3:] == expected_perms

def check_service_running(service):
    try:
        result = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except:
        return False