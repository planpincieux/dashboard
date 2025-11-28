FROM python:3.12-slim

# system deps required to build some python packages and to run GDAL/Postgres clients
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    postgresql-client \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    btop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_LIBRARY_PATH=/usr/lib/libgdal.so \
    GUNICORN_WORKERS=3

# Set work directory
WORKDIR /ppcx

# install python deps from pyproject.toml (pip will build via PEP517)
COPY pyproject.toml pyproject.toml
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir .

# Copy project files and   Set the working directory to the Django app
COPY ./app ./app
WORKDIR /ppcx/app

EXPOSE 8000

# Start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]