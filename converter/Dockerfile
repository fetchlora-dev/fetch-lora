# Fetch LoRA → Draw Things converter (GPL-3.0, see LICENSE in this folder).
# Stage 1 builds Draw Things' own LoRAConverter from draw-things-community (GPL-3.0).
# Stage 2 is a small runtime that watches the trainer's export folder.
FROM swift:6.4.2-jammy AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      clang llvm lld git curl ca-certificates python3 \
      libpng-dev libjpeg-dev libatlas-base-dev libblas-dev libsqlite3-dev libomp-14-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel
ARG DT_COMMIT=4e986724c6c65cdc8b9eeb8a2d2f4775c698e007
WORKDIR /src
RUN git init draw-things-community && cd draw-things-community \
    && git remote add origin https://github.com/drawthingsai/draw-things-community.git \
    && git fetch --depth 1 origin ${DT_COMMIT} && git checkout FETCH_HEAD
COPY build.sh /build.sh
RUN --mount=type=cache,target=/cache mkdir -p /out && bash /build.sh

FROM swift:6.4.2-jammy-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libatlas3-base libpng16-16 libjpeg8 libomp5-14 zlib1g python3 \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib/llvm-14/lib
COPY --from=build /out/LoRAConverter /out/SOURCE_COMMIT.txt /opt/dt/
COPY watch.py /opt/dt/watch.py
CMD ["python3", "-u", "/opt/dt/watch.py"]
