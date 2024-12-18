#!/bin/bash

# Set variables
IMAGE_NAME="orkunispir/inference-service"
TAG="latest"
POD_NAME="inference-pod"
NAMESPACE="modeling"
POD_YAML="gpu-pod.yaml"

# 1. Build the Docker image
echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build -t "$IMAGE_NAME:$TAG" .

# 2. Push the Docker image to the registry
echo "Pushing Docker image: $IMAGE_NAME:$TAG"
docker push "$IMAGE_NAME:$TAG"

# 3. Delete the existing pod (if it exists)
echo "Deleting pod: $POD_NAME in namespace: $NAMESPACE"
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --grace-period=0 --force

# 4. Apply the updated pod YAML
echo "Applying pod YAML: $POD_YAML"
kubectl apply -f "$POD_YAML" -n "$NAMESPACE"

echo "Deployment complete!"