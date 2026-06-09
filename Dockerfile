FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY .env .
EXPOSE 8050
HEALTHCHECK CMD curl --fail http://localhost:8050/ || exit 1
ENTRYPOINT ["python", "app.py"]