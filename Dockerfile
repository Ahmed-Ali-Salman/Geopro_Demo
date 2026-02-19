# GeoPro Demo - Dockerfile
# Public Image: salmaniv/geopro-demo:v2.2.1
# https://hub.docker.com/r/salmaniv/geopro-demo

FROM continuumio/miniconda3

WORKDIR /app

# Use Conda for base env
RUN conda install -y -c conda-forge \
    python=3.10 \
    gdal \
    libgdal \
    open3d \
    && conda clean -afy

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .
RUN pip install --no-cache-dir -e .

# Default command: launch the UI
EXPOSE 8000
CMD ["geopro", "serve", "--config", "config/default.yaml"]
