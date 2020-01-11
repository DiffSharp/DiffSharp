FROM mcr.microsoft.com/dotnet/core/sdk:3.1
WORKDIR /code/DiffSharp
COPY . /code/DiffSharp
RUN dotnet build

