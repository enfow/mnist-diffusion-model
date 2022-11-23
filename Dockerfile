FROM python:3.10.8-buster

WORKDIR /tmp
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

CMD ["python"]
