# MLOps on Kubernetes Project
## Overview
This repository contains a complete MLOps infrastructure implementation using Kubernetes. It provides an end-to-end solution for model training, deployment, serving, and monitoring in a scalable, reproducible environment.
## Features
- **Model Training Pipeline**: Automated training jobs with persistent storage
- **Model Serving**: REST API for real-time predictions
- **CI/CD Integration**: Automated testing and deployment
- **Horizontal Scaling**: Automatically scale based on demand
- **Observability**: Monitoring, logging, and alerting
- **Infrastructure as Code**: All infrastructure defined as Kubernetes manifests

## Prerequisites
- Kubernetes cluster (v1.20+)
- kubectl configured to access your cluster
- Docker (20.10+)
- Helm (v3.0+)
- Python 3.9+

## Architecture
The project consists of the following components:

1. **Training Pipeline**: Batch jobs for model training and evaluation
2. **Model Registry**: Storage for model artifacts and metadata
3. **Model Serving**: REST API for real-time predictions
4. **Monitoring**: Prometheus and Grafana dashboards

## Setup Instructions
### 1. Clone the Reporsitory
```bash
git clone https://github.com/yamalies/mlops-k8s-project.git
cd mlops-k8s-project
```
### 2. Build and Push Docker image
```bash
# Build and push model training image
docker build -t yamal50000/model-trainer:latest ./model
docker push yamal50000/model-trainer:latest

# Build and push API image
docker build -t yamal50000/ml-model-api:latest ./api
docker push yamal50000/ml-model-api:latest
```
### 3. Deploy Persistent Storage
```bash
kubectl apply -f k8s/pv-pvc.yaml
```
### 4. Deploy Model API
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```
### 5. Run Training Job
```bash
kubectl apply -f k8s/batch-job.yaml
```
### 6. Set up Monitoring
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
kubectl apply -f k8s/service-monitor.yaml
```
## Usage Examples
### Check the Status of Your Deployments
```bash
kubectl get deployments
kubectl get pods
```
### View API logs
```bash
kubectl logs -f deployment/ml-model-api
```
### Make a prediction Request
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  http://model-api.example.com/predict
```
### Create a Model Training Job
```bash
kubectl create job --from=cronjob/model-retraining manual-training-job
```