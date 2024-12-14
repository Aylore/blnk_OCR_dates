# Use the official Python image as a base
FROM python:3.10-slim
FROM tensorflow/tensorflow:2.12.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Start the server
CMD make run
#["python", "manage.py", "runserver", "0.0.0.0:8000"]
