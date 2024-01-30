FROM python:3.10-slim

WORKDIR /app

COPY requirements ./

RUN pip install --no-cache-dir -r requirements

COPY script.sh .

RUN ./script.sh

COPY ./app/ .

COPY ./model /model

COPY ./.cdsapirc /root/

ENV TZ=Europe/Warsaw

CMD ["python", "main.py"]