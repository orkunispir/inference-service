# Use an official NVIDIA CUDA image as the base

# Set working directory
WORKDIR /app

# Copy your application code
COPY app/ /app/

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your FastAPI app runs on
EXPOSE 80

# Command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"] 