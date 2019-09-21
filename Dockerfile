FROM mcr.microsoft.com/dotnet/core/sdk:2.2
WORKDIR /code/DiffSharp
COPY . /code/DiffSharp
RUN dotnet build

