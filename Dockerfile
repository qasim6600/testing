
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .
# Set environment variable for Groq (or handle with a secure method)
ENV GROQ_API_KEY=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
ENV GRADIO_SERVER_PORT=80
ENV GRADIO_SERVER_NAME=0.0.0.0
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

