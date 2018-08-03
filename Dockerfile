FROM microsoft/dotnet:sdk AS build-env

RUN apt-get update
RUN apt-get install -y libopenblas-dev

# Copy everything and build
COPY . ./
RUN ./build.sh

