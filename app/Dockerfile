FROM python:3.6-alpine3.7

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev

WORKDIR /src
COPY ./app/requirements.txt /src/requirements.txt
RUN pip install --upgrade pip
RUN apk add build-base
RUN pip3 install -r /src/requirements.txt
RUN echo "Hello"

COPY ./app .
EXPOSE 5000

CMD ["python3", "/app/main.py"]