FROM microsoft/dotnet:sdk AS build-env

# Copy everything and build
COPY . ./
RUN build.sh

