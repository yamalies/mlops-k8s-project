FROM python:latest

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY train.py model.py ./
COPY /data/train.csv /data/train.csv

# Create data directory
RUN mkdir -p /data

# Default command runs training
ENTRYPOINT ["python", "train.py"]
