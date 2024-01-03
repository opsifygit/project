FROM python:3.10-alpine AS builder

WORKDIR /src
COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
RUN echo "Hello"

COPY . .
EXPOSE 5000

CMD ["python3", "main.py"]