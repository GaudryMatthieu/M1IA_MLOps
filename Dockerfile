FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]