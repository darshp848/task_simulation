# Headless Hill Climb sim for the home cluster (CPU only): MuJoCo + Dream's yam_mujoco example + the harness.
# Rendering uses Mesa: EGL on the Inspiron's Intel iGPU (/dev/dri), OSMesa on nodes without one.
#   docker build -t hill-climb-sim -f cluster/Dockerfile cluster
#   docker run --rm -v $PWD:/repo -w /repo --device /dev/dri -e MUJOCO_GL=egl hill-climb-sim python -m pytest -q
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgl1-mesa-dri libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "dreamlabs==1.0.37" mujoco numpy pillow pyyaml pytest
ENV MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
