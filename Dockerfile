# Base Image
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        wget \
        ca-certificates

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install uvicorn fastapi

# Copy application code (assuming your application code is in a directory named 'src')
COPY app .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using uvicorn
# Assuming your main FastAPI app is in src/main.py and the app instance is named 'app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]