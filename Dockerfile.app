FROM python:3.10

# requirements for app are installed
COPY ./requirements.app /tmp/requirements.app
RUN pip3 install --upgrade pip
RUN pip install -r /tmp/requirements.app

# run backend app
WORKDIR /app
EXPOSE $INFERENCE_MICROSERVICE_PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port $INFERENCE_MICROSERVICE_PORT --reload