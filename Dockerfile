# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement files first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all other files
COPY . .

# Expose port (Render uses 10000 by default)
EXPOSE 10000

# Command to run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
