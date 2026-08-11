FROM python:3.12-slim
WORKDIR /app
COPY src/gpu_lab/fixed_relay.py /app/fixed_relay.py
USER 10003:10003
CMD ["python", "/app/fixed_relay.py"]
