#!/bin/bash

set -e

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-ap-south-1}
REGISTRY=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY

docker build -f backend.dockerfile -t cognitiveai-backend:latest .
docker tag cognitiveai-backend:latest $REGISTRY/cognitiveai-backend:latest
docker push $REGISTRY/cognitiveai-backend:latest

docker build -f frontend.dockerfile -t cognitiveai-frontend:latest .
docker tag cognitiveai-frontend:latest $REGISTRY/cognitiveai-frontend:latest
docker push $REGISTRY/cognitiveai-frontend:latest

kubectl apply -k k8s/

echo "Deployment complete. Services:"
kubectl -n cognitiveai get svc
