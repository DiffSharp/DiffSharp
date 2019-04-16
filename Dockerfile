FROM mcr.microsoft.com/dotnet/core/sdk:2.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        bsdtar \
        build-essential &&\
    rm -rf /var/lib/apt/lists/*

# Install libtorch
RUN mkdir /code
RUN wget -qO- https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip | bsdtar -xvf- -C /code
ENV LD_LIBRARY_PATH /code/libtorch/lib

RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.2/cmake-3.14.2-Linux-x86_64.sh && sh cmake-3.14.2-Linux-x86_64.sh --skip-license

# Build libTorchSharp
RUN cd /code && git clone https://github.com/interesaaat/LibTorchSharp.git && cd LibTorchSharp && git checkout b870e00 && cmake -DTorch_DIR=/code/libtorch/share/cmake/Torch . && make

# Build DiffSharp
COPY . /code/DiffSharp
RUN cd /code/DiffSharp && sh build.sh
