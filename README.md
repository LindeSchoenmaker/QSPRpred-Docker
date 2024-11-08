# QSPRpred-Docker

This is a fork of the QSPRpred-Docker developed by the VHP4Safety project. This basis is used to provide QSAR models for Mechpath.

This GitHub repo is set up for the development of a Docker-based service on the QSPRpred tool [github.com/CDDLeiden/QSPRpred](https://github.com/CDDLeiden/QSPRpred). In the initial stage, the service will have prediction functionalities based on imported, pre-trained model(s). 

## Flask Application Deployment with Docker
[![Docker Image CI](https://github.com/VHP4Safety/QSPRpred-Docker/actions/workflows/docker-image.yml/badge.svg)](https://github.com/VHP4Safety/QSPRpred-Docker/actions/workflows/docker-image.yml)

This guide covers the setup and usage of a Flask application for predictions via a Docker container. The application provides both a user interface for interacting with the model and an API for programmatic access.

### Building the Docker Image
To build the Docker image for the Flask application, use the following command:
```sh
docker build -t qspr_flask_image -f Dockerfile .
```
### Running the Docker Container
Run the Docker container with the following command. This command also mounts a local directory to the container to make model files accessible:
```sh
docker run -d -p 5000:5000 --name qspr_flask_container -v $(pwd)/models:/usr/src/app/models  qspr_flask_image
```

### Accessing the Application
Once the container is running, navigate to http://localhost:5000 in your web browser. You will see the Flask application interface which allows you to:

- Select one or more prediction models.
- Input multiple SMILES strings either through a text box or by uploading a CSV file (e.g. the [smiles_sample.csv](smiles_sample.csv)).
- Download the prediction results in various formats, or generate a report.

![QSPRpred UI](qsprpred-ui.png?raw=true "UI")

### Using the API
You can also interact with the Flask application via its API from your coding environment. Below are examples of how to use the API:

#### Example: JSON Output
To initiate a prediction and receive the results in JSON format, use:
```sh
curl -X POST localhost:5000/api     -H "Content-Type: application/json"     -d '{
        "smiles": ["C1=CC=CC=C1C(=O)NC2=CC=CC=C2", "CC(=O)OC1=CC=CC=C1C(=O)O"],
        "models": ["P10827_RF_Model", "P10828_RF_Model"],
        "format": "text"
    }'
```
#### Example: CSV Output
To receive the prediction results as a CSV file, use:
```sh
curl -X POST localhost:5000/api     -H "Content-Type: application/json"     -d '{
        "smiles": ["C1=CC=CC=C1C(=O)NC2=CC=CC=C2", "CC(=O)OC1=CC=CC=C1C(=O)O"],
        "models": ["P10827_RF_Model", "P10828_RF_Model"],
        "format": "csv"
    }' -o predictions.csv
```

