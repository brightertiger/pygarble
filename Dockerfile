FROM python:3.12-slim
WORKDIR /src
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY pygarble/ pygarble/
RUN pip install --no-cache-dir . && rm -rf /src/*
WORKDIR /
CMD ["python"]
