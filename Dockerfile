# Use official Python 3.12 slim image for a lightweight, secure base
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for compiling extensions (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the actual source code and tests into the container
COPY cfdlite/ cfdlite/
COPY tests/ tests/

# Create a non-root user for security best practices
RUN useradd -m cfduser
RUN chown -R cfduser:cfduser /app
USER cfduser

# Set the default command to run the test suite to verify the build works
CMD ["python", "-m", "unittest", "discover", "tests"]
