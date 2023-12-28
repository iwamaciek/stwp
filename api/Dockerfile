FROM python:3.10-slim

WORKDIR /app

COPY requirements ./

RUN pip install --no-cache-dir -r requirements

COPY ./app/ .

EXPOSE 8888

CMD ["python", "main.py"]