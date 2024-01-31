<img src="https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png" width="400" alt="Alt Text">

# Transformer_pytorch NLP Project

This project implements a Transformer model for natural language processing (NLP) tasks using PyTorch. The Transformer is trained on a custom date translation dataset.

## Project Structure

The project has the following directory structure:

- `artifacts`: Contains model artifacts and data files.
  - `source.txt`: Source sentences dataset.
  - `translation.txt`: Translation sentences dataset.
  - `transformer_model.pth`: Pre-trained Transformer model.
- `ebextensions`: Elastic Beanstalk configurations.
- `logs`: Log files generated during the training and deployment.
- `notebook`: Jupyter notebook for Transformer implementation (`Transformer.ipynb`).
- `src`: Source code directory.
  - `__init__.py`: Initialization file for the source code.
  - `application.py`: Application-specific code.
  - `components`: Components used in the project.
    - `__init__.py`: Initialization file for components.
    - `data_ingestion.py`: Data ingestion module.
    - `model_trainer.py`: Module for training the Transformer model.
  - `exception.py`: Custom exception handling module.
  - `logger.py`: Logging module.
  - `model`: Directory for model-related code.
    - `__init__.py`: Initialization file for model.
    - `Transformer.py`: Implementation of the Transformer model.
  - `pipeline`: Directory for processing pipelines.
    - `__init__.py`: Initialization file for pipelines.
    - `predict_pipeline.py`: Pipeline for making predictions using the trained model.
  - `templates`: HTML templates.
    - `index.html`: Default template for the web application.
  - `utils.py`: Utility functions used across the project.
- `requirements.txt`: List of project dependencies.
- `setup.py`: Setup script for installing the project.

## Transformer Implementation

The `Transformer.py` file in the `src/model` directory contains the implementation of the Transformer model. It includes modules for encoding, decoding, multi-head attention, layer normalization, and more.

## Data Ingestion

The `data_ingestion.py` module in the `src/components` directory generates and loads the date translation dataset. It uses Faker and Babel libraries to create a synthetic dataset of human-readable and machine-readable date representations.

## Model Training

The `model_trainer.py` module in the `src/components` directory trains the Transformer model using the generated date translation dataset. It includes code for data preprocessing, model initialization, training loop, and evaluation.

## Notebooks

The `notebook/Transformer.ipynb` Jupyter notebook provides a detailed walkthrough of the Transformer model implementation and training process.

## Deployment

The project can be deployed on AWS Elastic Beanstalk using the provided configurations in the `ebextensions` directory. The deployment process can be managed through AWS CodePipeline.

## Requirements

Make sure to install the required dependencies by running:

```bash
pip install -r requirements.txt
