# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Define the command to run the application
# This will process all *.md files in the /app/data directory by default
# Users can override this in compose.yaml or by passing arguments to `docker-compose run`
CMD ["python", "entigraph.py"]
