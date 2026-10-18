FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=2
ARG G2OPY_REV=5587024b17fd812c66d91740716fbf0bf5824fbc
ARG PANGOLIN_REV=3ac794aff96c3db103ec2bbc298ab013eaf6f6e8
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/src
ENV QT_X11_NO_MITSHM=1 LIBGL_ALWAYS_SOFTWARE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build build-essential pkg-config \
    python3 python3-dev python3-pip python3-tk \
    libeigen3-dev libsuitesparse-dev libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    libx11-dev libxi-dev libxxf86vm-dev libxrandr-dev libjpeg-dev libpng-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1-mesa-dri \
    libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/native
RUN git clone https://github.com/uoip/g2opy.git \
    && git -C g2opy checkout "$G2OPY_REV" \
    && git clone https://github.com/uoip/pangolin.git \
    && git -C pangolin checkout "$PANGOLIN_REV" \
    && git clone --depth 1 --branch v2.11.1 https://github.com/pybind/pybind11.git \
    && rm -rf g2opy/EXTERNAL/pybind11 pangolin/external/pybind11 \
    && cp -a pybind11 g2opy/EXTERNAL/pybind11 \
    && cp -a pybind11 pangolin/external/pybind11
COPY docker/patch_bindings.py /opt/native/patch_bindings.py
RUN python3 patch_bindings.py \
    && cmake -S g2opy -B g2opy/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_WITH_MARCH_NATIVE=OFF -DG2O_USE_OPENGL=OFF \
    && cmake --build g2opy/build --target g2o -j "$BUILD_JOBS" \
    && mkdir -p /usr/local/lib/python3.10/dist-packages \
    && cp g2opy/lib/g2o*.so /usr/local/lib/python3.10/dist-packages/

RUN cmake -S pangolin -B pangolin/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
       -DBUILD_SHARED_LIBS=OFF \
       -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
    && cmake --build pangolin/build -j "$BUILD_JOBS" \
    && cp pangolin/pangolin*.so /usr/local/lib/python3.10/dist-packages/ \
    && ldconfig

COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install --no-cache-dir 'pip<26' 'setuptools<76' wheel \
    && python3 -m pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir -r /opt/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/finalize_bindings.py /opt/native/finalize_bindings.py
RUN python3 /opt/native/finalize_bindings.py \
    && python3 -c "import g2o, pangolin, cv2, open3d, tkinter"
WORKDIR /workspace
COPY . .
RUN python3 -c "from fetal_slam.features.modules.lighterglue import LighterGlue; LighterGlue()" \
    && python3 -m pytest -q
ENTRYPOINT ["bash", "/workspace/docker/entrypoint.sh"]
CMD ["unity"]
