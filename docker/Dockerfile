FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools
    git \
    curl \
    wget \
    ca-certificates \
    # Image processing dependencies
    libgl1 \
    libglib2.0-0 \
    # Utilities
    zip \
    unzip \
    procps \
    nano \
    gosu \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

ARG WORKSPACE=/workspace

ARG USER_UID
ARG USER_GID
ARG USER_NAME
ARG GROUP_NAME=${USER_NAME}

ENV HOME=/home/${USER_NAME}

RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd --gid ${USER_GID} --non-unique ${GROUP_NAME} \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} --groups sudo --base-dir /home --shell /bin/bash ${USER_NAME} \
    && touch ${HOME}/.sudo_as_admin_successful \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} /home

USER ${USER_NAME}
ARG REPO_NAME
ENV WORK_DIR=${HOME}/workspace/${REPO_NAME}
ENV PYTHON_VERSION=3.14.2 \
    UV_PROJECT_ENVIRONMENT=${HOME}/.venv

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${HOME}/.local/bin:${PATH}"

COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock README.md ./
COPY --chown=${USER_UID}:${USER_GID} src ./src

RUN uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${UV_PROJECT_ENVIRONMENT} \
    && uv sync \
    && uv clean

# Default command
CMD ["bash"]

USER root
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
