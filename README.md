# Adaptive Binary Classifier Implementation

## Project Description

This repository is a dockerized implementation of the Random Forest binary classifier. It is implemented in flexible way that it can be used with any binary classification dataset with the use of JSON-formatted data schema file.

The main purpose of this repository is to provide a complete example of a machine learning model implementation that is ready for deployment.

Here are the highlights of this implementation:

- Implementation contains a full-suite of functionality such as:
  - A standardized project structure
  - A generalized data schema structure to be used for binary classification datasets
  - A flexible preprocessing pipeline built using **SciKit-Learn** and **feature-engine**
  - A Random Forest algorithm built using **SciKit-Learn**
  - Hyperparameter-tuning using **scikit-optimize**
  - SHAP explainer using the **shap** package
  - **FASTAPI** inference service which provides endpoints for predictions and local explanations.
  - **Pydantic** data validation is used for the schema, training and test files, as well as the inference request data.
  - Error handling and logging using **Python's logging** module.
  - Comprehensive set of unit, integration, coverage and performance tests using **pytest** and **pytest-cov**.
  - Static code analysis using **flake8**, **black**, **isort**, **safety**, and **radon**.
  - Test automation using **tox**.
- It is flexible enough to be used with any binary classification dataset without requiring any code change albeit with the following restrictions:
  - The data must be in CSV format
  - The number of rows must not exceed 20,000. Number of columns must not exceed 200. The model may function with larger datasets, but it has not been performance tested on larger datasets.
  - Features must be one of the following two types: NUMERIC or CATEGORICAL. Other data types are not supported. Note that CATEGORICAL type includes boolean type
  - The train and test (or prediction) files must contain an ID field. The train data must also contain a target field.
- The data need not be preprocessed because the implementation already contains logic to handle missing values, categorical features, outliers, and scaling.

This repository is part of a tutorial series called [Creating Adaptive ML Models](https://docs.readytensor.ai/category/creating-adaptive-ml-models) on Ready Tensor, a web platform for AI developers and users. The purpose of the tutorial series is to help AI developers create adaptive algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

The tutorial series is divided into 3 modules as follows:

1. **Model development**: This module covers the following tutorials:

   - **Creating Standardized Project Structure**: This tutorial reviews the standardized project structure we will use in this series.
   - **Using Data Schemas**: This tutorial discusses how data schemas can be used to avoid hard-coding implementations to specific datasets.
   - **Creating Adaptive Data Preprocessing-Pipelines**: In this tutorial, we cover how to create data preprocessing pipelines that can work with diverse set of algorithms such as tree-based models and neural networks.
   - **Building a Binary Classifier in Python**: This tutorial reviews a binary classifier model implementation which enables providing a common interface for various types of algorithms.
   - **Hyper-Parameter Tuning (HPT) Using SciKit-Optimize**: This tutorial covers how to use SciKit-Optimize to perform hyper-parameter tuning on ML models.
   - **Integrating FastAPI for Online Inference**: This tutorial covers how we can set up an inference service using FastAPI to provide online predictions from our machine learning model.
   - **Model Interpretability with Shapley Values**: This tutorial describes implementing an eXplainable AI (XAI) technique called Shap to provide interpretability to ML models.

2. **Quality Assurance for ML models**: This module covers the following tutorials:

   - **Error Handling and Logging**: In this tutorial, we review how to add basic error handling and logging to ML model implementations.
   - **Data Validation Using Pydantic**: This tutorial covers how we can use the pydantic library to validate input data for our machine learning model implementation.
   - **Testing ML Model Implementations**: This 6-part comprehensive tutorial covers the topic of testing machine learning model implementations including unit, integration, coverage and performance testing.
   - **Static Code Analysis for ML Models**: This is a two-part tutorial introducing static code analysis, discussing its importance in machine learning projects, and demonstrating how to implement it using various Python tools while automating the process with Tox.

3. **Model Containerization for Deployment**: This module covers the following tutorials:

   - **Overview of ML Model Containerization**: In this tutorial, we review why containerization is a key part of deployment of ML models and explore its benefits. We also review common containerization patterns considering whether the container includes training, batch prediction, or inference services.
   - **Containerizing ML Models - The Hybrid Pattern**: In this tutorial, we cover how to containerize an ML model with Docker to include both training and inference services.

This particular branch called [module-3-complete](https://github.com/readytensor/adaptive_binary_classifier/tree/module-1-complete) is the completion point of third (and final) module in the series. This is the final, completion version of the code for the binary classifier model.

## Project Structure

```txt
adaptive_binary_classifier/
├── examples/
├── model_inputs_outputs/
│   ├── inputs/
│   │   ├── data/
│   │   │   ├── testing/
│   │   │   └── training/
│   │   └── schema/
│   ├── model/
│   │   └── artifacts/
│   └── outputs/
│       ├── errors/
│       ├── hpt_outputs/
│       └── predictions/
├── requirements/
│   ├── requirements.txt
│   └── requirements_text.txt
├── src/
│   ├── config/
│   ├── data_models/
│   ├── hyperparameter_tuning/
│   ├── prediction/
│   ├── preprocessing/
│   ├── schema/
│   └── xai/
├── tests/
│   ├── integration_tests/
│   ├── performance_tests/
│   ├── test_resources/
│   ├── test_results/
│   │   ├── coverage_tests/
│   │   └── performance_tests/
│   └── unit_tests/
│       ├── (mirrors /src structure)
│       └── ...
├── tmp/
├── .dockerignore
├── .gitignore
├── docker-compose.yaml
├── Dockerfile
├── entrypoint.sh
├── LICENSE
├── pytest.ini
├── README.md
└── tox.ini
```

- **`examples/`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. This directory is further divided into:
  - **`/inputs/`**: This directory contains all the input files for this project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts/`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs/`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results.
- **`requirements/`**: This directory contains the requirements files. We have multiple requirements files for different purposes:
  - `requirements.txt` for the main code in the `src` directory
  - `requirements_text.txt` for dependencies required to run tests in the `tests` directory.
  - `requirements_quality.txt` for dependencies related to formatting and style checks.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories :
  - **`config/`**: for configuration files such as hyperparameters, model configuration, hyperparameter tuning-configuration specs, paths, and preprocessing configuration.
  - **`data_models/`**: for data models for input validation including the schema, training and test files, and the inference request data.
  - **`hyperparameter_tuning/`**: for hyperparameter-tuning (HPT) functionality using scikit-optimize
  - **`prediction/`**: for prediction model (random forest) script
  - **`preprocessing/`**: for data preprocessing scripts including the preprocessing pipeline, target encoder, and custom transformers.
  - **`schema/`**: for schema handler script.
  - **`xai/`**: explainable AI scripts (SHAP explainer)
    Furthermore, we have the following scripts in `src`:
  - **`logger.py`**: This script contains the logger configuration.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`serve.py`**: This script is used to serve the model as a REST API. It loads the artifacts and creates a FastAPI server to serve the model. It provides 3 endpoints: `/ping`, `/infer`, and `/explain`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions. The `/explain` endpoint is used to get local explanations for the predictions.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`. It also saves a SHAP explainer object in the path `./model/artifacts/`. When the train task is run with a flag to perform hyperparameter tuning, it also saves the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`tests/`**: This directory contains all the tests for the project and associated resources and results.
  - **`integration_tests/`**: This directory contains the integration tests for the project. We cover four main workflows: data preprocessing, training, prediction, and inference service.
  - **`performance_tests/`**: This directory contains performance tests for the training and batch prediction workflows in the script `test_train_predict.py`. It also contains performance tests for the inference service workflow in the script `test_inference_apis.py`. Helper functions are defined in the script `performance_test_helpers.py`. Fixtures and other setup are contained in the script `conftest.py`.
  - **`test_resources/`**: This folder contains various resources needed in the tests, such as trained model artifacts (including the preprocessing pipeline, target encoder, explainer, etc.). These resources are used in integration tests and performance tests.
  - **`test_results/`**: This folder contains the results for the performance tests. These are persisted to disk for later analysis.
  - **`unit_tests/`**: This folder contains all the unit tests for the project. It is further divided into subdirectories mirroring the structure of the `src` folder. Each subdirectory contains unit tests for the corresponding script in the `src` folder.
- **`tmp/`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.dockerignore`**: This file specifies the files and folders that should be ignored by Docker.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`docker-compose.yaml`**: This file is used to define the services that make up the application. It is used by Docker Compose to run the application.
- **`Dockerfile`**: This file is used to build the Docker image for the application.
- **`entry_point.sh`**: This file is used as the entry point for the Docker container. It is used to run the application. When the container is run using one of the commands `train`, `predict` or `serve`, this script runs the corresponding script in the `src` folder to execute the task.
- **`LICENSE`**: This file contains the license for the project.
- **`pytest.ini`**: This is the configuration file for pytest, the testing framework used in this project.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`tox.ini`**: This is the configuration file for tox, the primary test runner used in this project.

## Usage

### Preparing your data

- If you plan to run this model implementation on your own binary classification dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template. You can also refer to the tutorial on [Using Data Schemas](https://docs.readytensor.ai/reference-materials/tutorials/adaptable-ml-models/using-schemas) for more information on how to create a schema file.

### To run locally (without Docker)

- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `requirements` directory.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`. If you want to run with hyperparameter tuning then include the `-t` flag. This will also save the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints. The service runs on port 8080.

### To run with Docker

1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t classifier_img .` <br/>
   Here `classifier_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction or inference service:

   - The train, batch predictions tasks and inference service tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete. When the inference service task is run, the container will keep running until you stop or kill it.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction or inference service tasks, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - The inference service runs on the container's port **8080**. Use the `-p` flag to map a port on local host to the port 8080 in the container.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.

4. You can run training with or without hyperparameter tuning:

   - To run training without hyperparameter tuning (i.e. using default hyperparameters), run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train` <br/>
     where `classifier_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
   - To run training with hyperparameter tuning, issue the command: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train -t` <br/>
     This will tune hyperparameters,and used the tuned hyperparameters to train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount. It will also save the hyperparameter tuning results in the `model_inputs_outputs/outputs/hpt_outputs` directory in the bind mount.

5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.

6. To run the inference service, issue the following command on the running container: <br/>
   `docker run -p 8080:8080 -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img serve` <br/>
   This starts the service on port 8080. You can query the service using the `/ping`, `/infer` and `/explain` endpoints. More information on the requests/responses on the endpoints is provided below.

## Using the Inference Service

### Getting Predictions

To get predictions for a single sample, use the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/infer
```

The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "<timestamp>",
  "requestId": "<uniquely generated id>",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.97548, 0.02452]
    }
  ]
}
```

### Getting predictions and local explanations

To get predictions and explanations for a single sample, use the following request to send to `/explain` endpoint (same structure as data for the `/infer` endpoint):

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/explain
```

The server will respond with a JSON object containing the predicted probabilities and locations for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-05-22T10:51:45.860800",
  "requestId": "0ed3d0b76d",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.92107, 0.07893],
      "explanation": {
        "baseline": [0.57775, 0.42225],
        "featureScores": {
          "Age_na": [0.05389, -0.05389],
          "Age": [0.02582, -0.02582],
          "SibSp": [-0.00469, 0.00469],
          "Parch": [0.00706, -0.00706],
          "Fare": [0.05561, -0.05561],
          "Embarked_S": [0.01582, -0.01582],
          "Embarked_C": [0.00393, -0.00393],
          "Embarked_Q": [0.00657, -0.00657],
          "Pclass_3": [0.0179, -0.0179],
          "Pclass_1": [0.02394, -0.02394],
          "Sex_male": [0.13747, -0.13747]
        }
      }
    }
  ],
  "explanationMethod": "Shap"
}
```

## OpenAPI

Since the service is implemented using FastAPI, we get automatic documentation of the APIs offered by the service. Visit the docs at `http://localhost:8080/docs`.

## Testing

### Running through Tox

To run the tests:
Tox is used for running the tests. To run the tests, simply run the command:

```bash
tox
```

This will run the tests as well as formatters `black` and `isort` and linter `flake8`. You can run tests corresponding to specific environment, or specific markers. Please check `tox.ini` file for configuration details.

### Running through Pytest

- Run the command `pytest` from the root directory of the repository.
- To run specific scripts, use the command `pytest <path_to_script>`.
- To run slow-running tests (which take longer to run): use the command `pytest -m slow`.

## Requirements

The requirements files are placed in the folder `requirements`.

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.

For testing, dependencies are listed in the file `requirements-test.txt`.

Dependencies for quality-tests are listed in the file `requirements-quality.txt`. You can install these packages by running the following command:

```python
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install -r requirements-quality.txt
```

Alternatively, you can let tox handle the installation of test dependencies for you. To do this, simply run the command `tox` from the root directory of the repository. This will create the environments, install dependencies, and run the tests as well as quality checks on the code.

## Contact Information

Repository created by Ready Tensor, Inc.
