FROM mcr.microsoft.com/dotnet/core/sdk:3.0
WORKDIR /code/DiffSharp
COPY . /code/DiffSharp
RUN dotnet build

