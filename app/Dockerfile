# Use an official Python runtime as the base image, and one that has CUDA support if you need it for the GPU
FROM nvcr.io/nvidia/pytorch:23.11-py3

# Set the working directory inside the container
WORKDIR /

# Copy the local project files to the container's working directory
COPY . /app

# Install the Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on (default is 8000)
EXPOSE 8000

# Define the command to run your FastAPI application using uvicorn
# Adjust the host and port if necessary.
# The --reload flag is useful for development, but you might want to remove it in production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]