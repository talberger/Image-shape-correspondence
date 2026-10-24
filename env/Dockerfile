# DISC main env for TAU Run:ai (GPU). Same versions as env/requirements_disc_gpu.txt.
# torch 1.12.1+cu113 / kaolin wheels bundle their own CUDA runtime -> no CUDA base image needed;
# the GPU driver comes from the cluster node.
# Build + push from the Mac (build context = this env/ folder only, so no data/code goes into the image):
#   cd ~/Projects/disc
#   docker buildx build --platform linux/amd64 -t tberger3/disc:v2 --push env/
# The image is PUBLIC on Docker Hub: never put code, data, keys or tokens in it.
FROM python:3.9.18-slim-bookworm

# system libs needed by open3d / opencv / pymeshlab, a C/C++ compiler (scikit-fmm builds from source),
# plus git for cloning the repo inside the workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_disc_gpu.txt /tmp/requirements_disc_gpu.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements_disc_gpu.txt \
    && rm /tmp/requirements_disc_gpu.txt

WORKDIR /workspace
CMD ["sleep", "infinity"]
