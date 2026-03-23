# CLRerNet lane detection environment
# Requires CUDA 12.1 compatible GPU for full functionality (NMS compilation + GPU inference)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    zip \
    unzip \
    procps \
    nano \
    gosu \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

ARG USER_UID
ARG USER_GID
ARG USER_NAME
ARG GROUP_NAME=${USER_NAME}
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9+PTX"

ENV HOME=/home/${USER_NAME}

RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd --gid ${USER_GID} --non-unique ${GROUP_NAME} \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} --groups sudo --base-dir /home --shell /bin/bash ${USER_NAME} \
    && touch ${HOME}/.sudo_as_admin_successful \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Clone CLRerNet repository (done as root before user switch)
ENV CLRERNET_ROOT=/opt/CLRerNet
RUN git clone --depth 1 https://github.com/hirotomusiker/CLRerNet.git ${CLRERNET_ROOT} \
    && chown -R ${USER_UID}:${USER_GID} ${CLRERNET_ROOT}

RUN mkdir -p /workspace \
    && mkdir -p /opt/venv \
    && chown -R ${USER_UID}:${USER_GID} /opt/venv \
    && chown -R ${USER_UID}:${USER_GID} /workspace \
    && chown -R ${USER_UID}:${USER_GID} /home

USER ${USER_NAME}
ARG REPO_NAME
ENV WORK_DIR=${HOME}/workspace/${REPO_NAME}
# Python 3.11.9 required by CLRerNet (mmlab stack constraint)
ENV PYTHON_VERSION=3.11.9 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONPATH="/opt/CLRerNet"
WORKDIR ${WORK_DIR}

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${HOME}/.local/bin:${PATH}"

COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock README.md ./

RUN uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${UV_PROJECT_ENVIRONMENT} \
    && uv sync --no-install-project \
    && uv clean

# Build CLRerNet custom NMS CUDA extension after dependencies are installed.
RUN cd ${CLRERNET_ROOT}/libs/models/layers/nms && python setup.py install

CMD ["bash"]

USER root
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
