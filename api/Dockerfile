FROM python:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model.pkl /model.pkl

EXPOSE 5000

CMD ["python", "app.py"]