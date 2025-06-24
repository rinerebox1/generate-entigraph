#!/bin/bash

set -euo pipefail

# --- Variables ---
IMAGE_NAME="entigraph_image:latest"
CONTAINER_NAME="entigraph_container"
DOCKERFILE_PATH="." # Path to the directory containing your Dockerfile

# --- Cleanup: Stop and Remove Existing Container ---
# We use `docker ps -q` to check if the container is running/exists.
# `|| true` is used to prevent the script from exiting if the container doesn't exist.
echo "Attempting to stop and remove Docker container: $CONTAINER_NAME..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
echo "Container cleanup complete."

# --- Cleanup: Remove Existing Image ---
# This ensures we build from a clean state.
# `|| true` handles the case where the image does not exist.
echo "Attempting to remove Docker image: $IMAGE_NAME..."
docker rmi $IMAGE_NAME || true
echo "Image cleanup complete."

# --- Build: Create New Image ---
echo "Building Docker image: $IMAGE_NAME from Dockerfile in $DOCKERFILE_PATH..."
# `--pull` ensures we have the latest base images for security and consistency.
docker build --pull -t $IMAGE_NAME $DOCKERFILE_PATH
# `set -e` will automatically handle the build failure, so the manual check is no longer strictly necessary,
# but can be kept for a more explicit error message.
echo "Docker image $IMAGE_NAME built successfully."

# --- Deploy: Run New Container ---
echo "Deploying container: $CONTAINER_NAME from image: $IMAGE_NAME..."
# Add any necessary port mappings or volume mounts here.
# `--rm` automatically removes the container when it exits.
# Example: docker run -d --rm -p 8080:80 --name $CONTAINER_NAME $IMAGE_NAME
docker run -d --rm --name $CONTAINER_NAME $IMAGE_NAME
echo "Container $CONTAINER_NAME deployed successfully."

# --- Verification ---
echo "Waiting a few seconds before showing logs..."
sleep 3
echo "--- Initial logs for $CONTAINER_NAME ---"
docker logs --tail 30 $CONTAINER_NAME
echo "--------------------------------------"

echo "Redeployment of $CONTAINER_NAME complete."
