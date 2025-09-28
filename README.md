# RoyalSecureAI

**Copyright © 2025 Devin B. Royal. All Rights Reserved.**

RoyalSecureAI is a collection of eight advanced algorithms designed to address critical challenges in privacy-preserving AI, cybersecurity, software installation, explainable AI, and DevOps automation. These algorithms are optimized for performance, cross-platform compatibility, and integration with modern frameworks, making them suitable for enterprise applications in healthcare, finance, IT, and IoT.

## Algorithms Overview

1. **Federated Learning Optimizer** (`federated_learning_optimizer.py`)
   - Trains AI models across distributed clients without sharing raw data, ensuring privacy.
   - Use Cases: Collaborative healthcare diagnostics, financial fraud detection.
   - Features: PyTorch/TensorFlow support, parallel processing, Snappy compression.

2. **Differential Privacy Engine** (`differential_privacy_engine.py`)
   - Adds Gaussian noise to gradients for privacy-preserving ML training.
   - Use Cases: Medical research, ad tech, regulatory compliance (GDPR).
   - Features: Framework-agnostic, vectorized operations, SciPy noise generation.

3. **Vulnerability Detection and Patching** (`vulnerability_patcher.py`)
   - Scans systems, isolates vulnerabilities, and applies patches in real time.
   - Use Cases: Enterprise IT, cloud security, IoT firmware protection.
   - Features: Cross-platform, integrates with scanners like ClamAV.

4. **Behavioral Anomaly Detector** (`anomaly_detector.py`)
   - Detects anomalies in system behavior using statistical modeling.
   - Use Cases: Network security, fraud detection, system monitoring.
   - Features: Sliding window, NumPy-optimized, no retraining needed.

5. **Post-Install Validation Algorithm** (`post_install_validator.py`)
   - Verifies file integrity, permissions, and service status post-installation.
   - Use Cases: Software deployment, CI/CD pipelines, system updates.
   - Features: SHA-256 checksums, cross-platform service checks.

6. **Rollback-Aware Installer Logic** (`rollback_installer.py`)
   - Executes installations with automatic rollback on failure.
   - Use Cases: Software installers, system upgrades, DevOps pipelines.
   - Features: Change tracking, robust error handling.

7. **Lightweight Explanation Algorithm** (`model_explainer.py`)
   - Generates human-readable explanations for AI model predictions.
   - Use Cases: Finance, healthcare, regulatory compliance.
   - Features: SHAP integration, framework-agnostic, fallback mode.

8. **Visual Explainer Engine** (`neural_visual_explainer.py`)
   - Visualizes neural network decision paths as heatmaps.
   - Use Cases: Model debugging, AI education, research.
   - Features: Matplotlib-based, PyTorch-optimized, base64 output.

## Installation

### Prerequisites
- **Hardware**: ≥8GB RAM, ≥4 CPU cores, 10GB free storage.
- **OS**: Windows, Linux, or macOS.
- **Python**: 3.8+ (install via [python.org](https://python.org) or Anaconda).
- **Dependencies**:
  ```bash
  pip install numpy scipy pandas torch tensorflow shap matplotlib networkx snappy
  ```
  - Optional: ClamAV (for vulnerability scanning), systemctl (Linux).
  - If dependencies fail, use fallbacks (e.g., skip `snappy`, mock ClamAV).

### Setup
1. Clone or download the project to a local directory (e.g., `RoyalSecureAI/`).
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies (see above).
4. Create test directories: `RoyalSecureAI/test_files/` and `RoyalSecureAI/install_dir/`.

## Usage

Each algorithm is a standalone Python script. Below are example commands to run each with synthetic test data.

### 1. Federated Learning Optimizer
```python
import torch.nn as nn
import numpy as np
from federated_learning_optimizer import federated_learning_optimizer, Client

clients = [Client(np.random.rand(100, 10), np.random.randint(0, 2, 100)) for _ in range(3)]
global_model = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))
result = federated_learning_optimizer(clients, global_model, rounds=2, learning_rate=0.01, framework='pytorch')
print("Final model:", result)
```

### 2. Differential Privacy Engine
```python
import torch
from differential_privacy_engine import differential_privacy_engine

data = [(torch.tensor(np.random.rand(32, 10), dtype=torch.float32), torch.tensor(np.random.randint(0, 2, 32), dtype=torch.long)) for _ in range(10)]
model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Softmax(dim=1))
result = differential_privacy_engine(model, data, epsilon=1.0, delta=1e-5, clip_norm=1.0, framework='pytorch')
print("Model:", result)
```

### 3. Vulnerability Detection and Patching
```python
import os
from vulnerability_patcher import vulnerability_patcher

os.makedirs("test_files", exist_ok=True)
with open("test_files/file1.txt", "w") as f:
    f.write("Vulnerable")
system_state = {"root_path": "test_files/"}
patch_database = {"file1.txt": "Patched content"}
vulnerability_patcher(system_state, patch_database)  # Modify to run once
```

### 4. Behavioral Anomaly Detector
```python
import numpy as np
from anomaly_detector import anomaly_detector

events = [{"id": i, "value": np.random.normal(0, 1) if i != 50 else 5.0} for i in range(100)]
model = anomaly_detector(events, window_size=10, threshold=3.0)
print("Model:", model)
```

### 5. Post-Install Validation
```python
import os, hashlib
from post_install_validator import post_install_validator

os.makedirs("install_dir", exist_ok=True)
with open("install_dir/app1.txt", "w") as f:
    f.write("Test")
expected_files = [{"name": "app1.txt", "checksum": hashlib.sha256(b"Test").hexdigest()}]
expected_permissions = {"app1.txt": "644"}
services = ["mock_service"]
results = post_install_validator("install_dir", expected_files, expected_permissions, services)
print("Results:", results)
```

### 6. Rollback-Aware Installer
```python
import os
from rollback_installer import rollback_aware_installer

os.makedirs("install_dir", exist_ok=True)
with open("source.txt", "w") as f:
    f.write("Source")
install_steps = [{"id": 1, "type": "file_copy", "source": "source.txt", "file": "dest.txt"}]
system_state = {"dir": "install_dir/"}
success, log = rollback_aware_installer(install_steps, system_state)
print("Success:", success, "Log:", log)
```

### 7. Lightweight Explanation
```python
import torch.nn as nn
import numpy as np
from model_explainer import explain_model_prediction

model = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))
input_data = np.random.rand(1, 10)
feature_names = [f"f{i}" for i in range(10)]
explanation = explain_model_prediction(model, input_data, feature_names)
print("Explanation:", explanation)
```

### 8. Visual Explainer
```python
import torch
from neural_visual_explainer import visual_explainer

model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
input_data = torch.tensor(np.random.rand(1, 10), dtype=torch.float32)
result = visual_explainer(model, input_data)
print("Visualization:", result["image"][:50], "...")
```

## Testing

To test all algorithms:
1. **Setup Environment**: Follow installation steps.
2. **Prepare Test Data**: Use synthetic data as shown in usage examples.
3. **Run Tests**: Execute each script with provided commands.
4. **Check Outputs**:
   - Verify logs in `RoyalSecureAI/logs/` for errors.
   - Confirm expected outcomes (e.g., model accuracy, file changes, anomaly alerts).
5. **Handle Issues**:
   - Missing libraries: Use fallbacks (e.g., no SHAP, mock ClamAV).
   - OS-specific: Code handles Windows/Linux differences.

**Expected Test Duration**: ~1-2 hours (setup: 15-30 min, testing: 5-15 min per algorithm).

## Performance
- **Federated Learning**: O(r * k * m * log k) (~ms/round for small models).
- **Differential Privacy**: O(b * m) (~ms/batch).
- **Vulnerability Patcher**: O(d) (~ms for small dirs).
- **Anomaly Detector**: O(w) (~ms/event).
- **Post-Install Validator**: O(f + s) (~ms).
- **Rollback Installer**: O(s) (~ms).
- **Lightweight Explanation**: O(s * f) (~ms without SHAP).
- **Visual Explainer**: O(l * n) (~ms).

## License
This project is proprietary software. See [LICENSE](LICENSE) for details.

## Contact
For inquiries, contact Devin B. Royal (details TBD). Unauthorized use is prohibited.
