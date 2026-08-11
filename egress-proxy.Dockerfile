FROM python:3.12-slim
WORKDIR /app
COPY src/gpu_lab/egress_proxy.py /app/egress_proxy.py
USER 10004:10004
CMD ["python", "/app/egress_proxy.py"]
