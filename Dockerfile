FROM zironycho/pytorch:1120-cpu-py38

WORKDIR /opt/src
#ENV GRADIO_SERVER_PORT 7860
COPY requirements.txt .

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

EXPOSE 80

ENTRYPOINT ["python3","inference.py"]