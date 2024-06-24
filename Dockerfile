# Use the official lightweight Python image.
FROM python:3.10.11-slim

# Set environment variables.
# Python won't try to write .pyc files on the container.
ENV PYTHONDONTWRITEBYTECODE 1
# Python outputs all messages directly to the terminal without buffering it first.
ENV PYTHONUNBUFFERED 1

# Set work directory.
WORKDIR /app

# Install system dependencies including CMake, which is required for dlib.
RUN apt-get update && apt-get install -y  \
    build-essential \
    cmake \
    pkg-config \
    g++ \
    gcc \
    libboost-python-dev \
    libgl1 \ 
    libglib2.0-0

# Install Python dependencies.
COPY requirements.txt /app/
RUN pip install numpy
RUN pip install cmake
RUN pip install dlib==19.24.0

RUN pip install --no-cache-dir -r requirements.txt


# Copy project.
COPY . /app/

# Expose the port the app runs on.
EXPOSE 5010

# Run the application:
CMD ["python3", "app.py"]
