# On-Device Training and Federated Learning Simulation for Energy Cost calculation

This is aimed at simulating the results for the metrics collected from the **On-Device Training** app.

---

## Features

1. **Firebase Data Processing**  
   - Filters and exports training metrics stored in a Firebase Realtime Database (recommended to use the same as in the On-Device training app).  
   - Extracts metrics based on specific models and thresholds.

2. **Federated Learning Simulation**  
   - Simulates a federated learning environment using Flower.  
   - Supports dynamic configuration of training rounds, client participation, and resource allocation.  

3. **Configurable Parameters**  
   - Unified configuration management with `config.yaml` for both Firebase and federated learning scripts.

---

## Prerequisites

1. **Python 3.8+**  
   Ensure Python is installed.  

2. **Install Dependencies**  
   Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Firebase Configuration**  
   - Place your Firebase service account credentials in `secrets/service-account.json`.  
   - Ensure the Firebase Realtime Database URL is defined in `config.yaml`.

4. **Configuration Files**  
   - Edit `config.yaml` to customize parameters.  
   - Prepare `device_config.json` to define device-specific configurations, by running the `firebase_retriever`.

---

## Usage

### 1. Firebase Data Processing
Filters and exports data from Firebase based on the specified `model_use` URL in `config.yaml`.

**Run the script:**
```bash
python simulation/firebase_retriver.py
```

**Output:**  
Filtered data will be exported to `deviceConfigurations.json`.

---

### 2. Federated Learning Simulation
Simulates federated learning with customizable parameters.

**Run the script:**
```bash
python simulation/simulation.py
```

**Configuration:**  
Define the following parameters in `config.yaml`:
- **`partitions`**: Number of federated clients.
- **`min_client_per_round`**: Minimum number of clients per round.
- **`backend_config`**: CPU configuration.
- **`gpu_backend_config`**: GPU configuration.

**Output:**  
Simulation results will be logged, and the training process will use the provided device configurations. To find the logs for a specific run follow the path `outputs/run-date/run-time`.

---

## Example Configuration Files

**`config.yaml`:**
```yaml
firebase_database_url: https://your-firebase-url.firebaseio.com/
model_use: https://example.com/path-to-your-model

partitions: 300
min_client_per_round: 2

backend_config:
  client_resources:
    num_cpus: 8
gpu_backend_config:
  client_resources: 
    num_gpus: 1
```

**`device_config.json`:**
```json
[
  {
    "device_id": "device_1",
    "model_link": "https://example.com/model_1.tflite",
    "energy_profile": 50
  },
  {
    "device_id": "device_2",
    "model_link": "https://example.com/model_2.tflite",
    "energy_profile": 75
  }
]
```

---

## Contributing

Feel free to contribute to this project by submitting pull requests, reporting issues, or suggesting enhancements.

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.
