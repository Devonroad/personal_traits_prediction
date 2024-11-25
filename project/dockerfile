FROM python:3.11-slim

WORKDIR /predict_app

COPY ["requirements.txt", "./"]

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ["model_wrapper.py", "predict.py", "model.bin", "./"]

EXPOSE 8185

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:8185", "predict:app" ]
