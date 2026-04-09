FROM ros:rolling-ros-core

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/poppy/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0
ENV LIBGL_ALWAYS_SOFTWARE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    xvfb \
    xauth \
    libgl1 \
    libglew2.2 \
    libglfw3 \
    libosmesa6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build

COPY requirements.txt /tmp/build/requirements.txt

RUN python3 -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && . "${VIRTUAL_ENV}/bin/activate" \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && python -m pip install -r /tmp/build/requirements.txt

WORKDIR /workspace

CMD ["bash"]
