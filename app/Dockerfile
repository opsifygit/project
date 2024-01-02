FROM python:3.10-alpine AS builder

WORKDIR /src
COPY requirements.txt /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python3", "main.py"]