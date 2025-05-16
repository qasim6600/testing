FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Set environment variable for Groq (or handle with a secure method)
# ENV GROQ_API_KEY=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
ENV GRADIO_SERVER_PORT=80
ENV GRADIO_SERVER_NAME=0.0.0.0
# Expose port used by Gradio
EXPOSE 80

# Run the app
CMD ["python", "app.py"]

