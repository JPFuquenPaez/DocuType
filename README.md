# Handwritten vs. Typed Document Classification System

## Introduction

ML pipeline for document image classification (handwritten vs. printed). It leverages HuggingFace open-source datasets (merci HuggingFace), several training strategies (transfer learning and fine-tuning), and MLflow for experiment tracking, model registry, and deployment. The system exposes a FastAPI web service for inference.


- **Image Classification**: Binary classification (handwritten vs. printed).
- **Open Data Integration**: Uses open-source datasets for both handwritten and printed text.
- **Experiment Management**: MLflow tracks all experiments, metrics, artifacts, and model versions.
- **Flexible Model Selection**: Register the best model globally, from all/selected experiments, or by run ID.
- **Model Registry & Promotion**: Automated registration and promotion to production using MLflow Model Registry and aliases.
- **FastAPI Web Service**: API for health checks and predictions.
- **Interpretability (Jupyter only)**: Grad-CAM and Occlusion for model explainability.
- **Dockerized Deployment**: Consistent, reproducible environments.

### Project Layout

```plaintext
YOUR_MAIN_DIRECTORY/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── webservice.py
│   ├── inference/
│   │   ├── inference_onnx.py
│   │   ├── inference.ipynb
│   │   ├── model_prod_details.py
│   │   ├── model_registry_prod_experiments.py
│   ├── training/
│   │   ├── training_experiments.ipynb
│   │   └── mlruns/
│   ├── data/
│   │   ├── data_import_handwritten_CORTO.py
│   │   ├── data_import_printed_handwritten_CATMUS.py
│   │   └── datasets/
│   └── webservice_predictions/
│       └── predictions.log
├── documentation/
│   └── docs/
│       └── index.md
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---


## Step-by-Step: Read Further Documentation with MkDocs

This project includes extended documentation built with [MkDocs](https://www.mkdocs.org/). To read the full documentation locally:

1. **Install MkDocs** (if not already installed):
   ```
   pip install mkdocs
   ```

2. **Serve the documentation locally** from the project root:
   ```
   mkdocs serve
   ```

OR go to documentation source which is in `documentation/docs/index.md` ==> command from terminal: python -m mkdocs serve

3. **Open your browser** and go to [http://127.0.0.1:8000](http://127.0.0.1:8000) to browse the documentation.

---

## Best Practices

Developed using [poetry](https://python-poetry.org/), which enables versioning, development of multiple services/libraries on the same machine via virtual environments, and dependency management by pinning package versions.

We use `ruff` for code formatting and to enforce our standards. We also use `mypy` for type checking.

The `lint_module.sh` script automates the application of these tools.

To run them and obtain a coverage report, use the `run_tests.sh` script.

Before proceeding, ensure you have the following installed on your machine:

- [poetry](https://python-poetry.org/)
- [docker](https://docs.docker.com/engine/install/)

You can then run:

```
poetry install
./lint_module.sh
./run_tests.sh
docker-compose up --build
```

---

## Dataset Details

Two datasets were explored, but CATMuS modern was selected as it contains examples for both classes. The single-class dataset was used for some examples and to test the model.

### Handwritten Text Images

- **Source**: [Corto-AI Handwritten Text Dataset](https://huggingface.co/datasets/corto-ai/handwritten-text)
- **Description**: Grayscale images of handwritten text, extracted from Parquet files and saved as PNG.
- **Processing**:
  - Split into `train`, `validation`, and `test`.
  - Label: `handwritten` (class `0`).
  - See [`data_import_handwritten_CORTO.py`](app/data/data_import_handwritten_CORTO.py) for preprocessing logic.

### Printed and Handwritten Text Images

- **Source**: [CATMuS Modern Dataset](https://huggingface.co/datasets/CATMuS/)
- **Description**: RGB images of both printed and handwritten text.
- **Processing**:
  - Organized by writing type (`handwritten` or `printed`).
  - Split into `train`, `validation`, and `test`.
  - Labels: `handwritten` (class `0`), `printed` (class `1`).
  - See [`data_import_printed_handwritten_CATMUS.py`](app/data/data_import_printed_handwritten_CATMUS.py).

- **Data Handling**:
  - All data is loaded and preprocessed using custom scripts.
  - Data augmentation and normalization are applied during training.

---

## Training Logic

### Model Architecture

Start by training your model with or w/o a Jupyter Notebook...

- **Base Model**: MobileNetV2 (pretrained on ImageNet).
- **Transfer Learning**:
  - Freeze convolutional and some other layers to retain pretrained features.
  - Replace classifier with a custom linear layer for binary classification.
- **Fine-Tuning**:
  - Optionally unfreeze specific last layers for further adaptation.
  - Multiple fine-tuned models can be trained with different layers, epochs, learning rates, or data splits.

### Training Process

1. **Data Augmentation**:
Could be improved.
   - Resize to `256x1024`.
   - Random rotations, color jitter, normalization.
2. **Subset Creation**:
   - Optionally set a subset (e.g., 10%) for rapid training and testing.
3. **Loss Function**:
   - Cross-Entropy Loss.
4. **Optimizer**:
   - Adam with `ReduceLROnPlateau` scheduler (monitors validation loss).
5. **Classification Metrics**:
   - Accuracy, Precision, Recall, F1 Score, ROC AUC, Confusion Matrix.
   - Metrics are logged per epoch and at test time.
6. **Logging**:
   - **MLflow** logs:
     - Parameters, metrics, artifacts (models, plots, transforms, sample datasets).
     - See [`training_experiments.ipynb`](app/training/training_experiments.ipynb) for full logic.
   - **Artifacts**:
     - PyTorch model (`.pth`), ONNX model, preprocessing transforms, sample datasets, plots.

### Experiment Tracking and Model Registry

- **MLflow Tracking**:
  - All experiments are tracked in [`mlruns/`](app/training/mlruns/). If it's your first time using MLflow please refer to documentation. You can use MLflow UI to interact easily with your experiments. 
  - Each run logs metrics, parameters, and artifacts.
  - Experiments can be created for different strategies (transfer learning, fine-tuning, etc.).

- **Model Selection & Registration**:
  - Use [`model_registry_prod_experiments.py`](app/inference/model_registry_prod_experiments.py) to:
    - Find the best run by a target metric (default: `validation_accuracy`) across all or selected experiments.
    - Optionally, select a specific experiment or run ID.
    - Register the best model(s) in the MLflow Model Registry.
    - Set tags (including dataset and model descriptions).
    - Promote the best version to production using the `prod` alias.
  - The logic supports:
    - **Global best**: Best model from all experiments.
    - **Experiment best**: Best model from a specific experiment.
    - **Manual selection**: Register a model by run ID.

- **Model Promotion**:
  - The latest/best version is assigned the `prod` alias.
  - Previous production models are archived.
  - See [`promote_to_production`](app/inference/model_registry_prod_experiments.py) for details.

- **Production Model Details**:
  - Use [`model_prod_details.py`](app/inference/model_prod_details.py) to inspect the current production model, including metrics, parameters, tags, and registration time.

---

## Inference & FastAPI Web Service

Using the provided codebase, you must build a system with [FastAPI](https://fastapi.tiangolo.com/) that allows:

- **Model Loading**:
  - The FastAPI service loads the production model from the MLflow Model Registry using the `prod` alias.
  - Preprocessing transforms are loaded from the corresponding artifact.

- **Endpoints**:
  - `/health`: Returns model status, version, and alias.
  - `/predict/`: Accepts an image file and returns the predicted class, confidence, model version, and alias.

- **Logging**:
  - All predictions are logged to `webservice_predictions/predictions.log` for traceability.

- **Error Handling**:
  - Robust error handling for model loading, prediction, and file logging.
  - See [`webservice.py`](app/webservice.py) and [`main.py`](app/main.py).

---

## Interpretability (Jupyter only)

Model interpretability is implemented in Jupyter notebooks for debugging and transparency. The idea is to trigger interpretability if prediction confidence is low (could be extended as a service).

- **Grad-CAM**: Visualizes important regions for each class using the last convolutional layer.
- **Occlusion**: Systematically occludes image regions to identify areas critical for classification.

---

## Experiment Management Workflow

1. **Train Models**:
   - Run experiments in [`training_experiments.ipynb`](app/training/training_experiments.ipynb).
   - Each run is tracked in MLflow with full metrics and artifacts.

2. **Select Best Model(s)**:
   - Use [`model_registry_prod_experiments.py`](app/inference/model_registry_prod_experiments.py) to:
     - Automatically select the best run by metric.
     - Register and promote to production.
     - Optionally, select by experiment or run ID.

3. **Inspect Production Model**:
   - Use [`model_prod_details.py`](app/inference/model_prod_details.py) to view details of the current production model.

4. **Deploy FastAPI Service**:
   - The service always loads the current production model.
   - Supports hot-swapping models by updating the `prod` alias in MLflow.

---

## Technical Requirements

- Use FastAPI to expose your service.
- Use Docker to containerize your application.
- You may use any machine learning library (scikit-learn, PyTorch, TensorFlow, etc.), but your `pyproject.toml` must be up-to-date and functional on a Linux system (debian/ubuntu).
- Compare the performance of at least 2 algorithms.
- Implement tests.

---

## Process

1. **Data Acquisition**: Use one or more public datasets, or generate them if necessary.
2. **Training & Testing**: Implement a system that allows training one or more algorithms on this data, evaluating performance, and returning evaluation metrics.
3. **Inference Service**: Implement an endpoint that accepts a request with data, runs it through an algorithm, and returns a result.
4. **Containerization**: Modify the Dockerfile as needed to include your dependencies. Ensure the service works with `docker compose up --build`.
5. **Testing**: Ensure everything is functional with tests.

---

## Execution

Before proceeding, ensure you have the following installed on your machine:

- [poetry](https://python-poetry.org/)
- [docker](https://docs.docker.com/engine/install/)

The `Dockerfile` and `docker-compose.yml` files allow you to launch the web service on port `9000`..

You can then run:

```
poetry install
./lint_module.sh
./run_tests.sh
docker-compose up --build
```

- Source code is in the `app/` directory.
- Everything needed to generate a virtual environment and run the application using poetry.
- Docker configuration files.
- Unit tests verifying your API's functionality.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ONNX Documentation](https://onnx.ai/)
- [Captum Documentation](https://captum.ai/docs/attribution_algorithms)
- [MkDocs Documentation](https://www.mkdocs.org/)

---