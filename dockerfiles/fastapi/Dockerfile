FROM python:3.12-slim

WORKDIR /app
COPY dockerfiles/fastapi/requirements.txt /requirements.txt
COPY dockerfiles/fastapi/app.py /app/app.py
COPY src /app/src

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8800
