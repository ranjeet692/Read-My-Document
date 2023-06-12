# Use an official Python runtime as the base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which the application will run (change it if necessary)
EXPOSE 8501

# Set the entrypoint command to start the Streamlit application
CMD ["streamlit", "run", "--server.port", "8501", "main.py"]
