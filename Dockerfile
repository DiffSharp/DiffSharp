FROM mcr.microsoft.com/dotnet/sdk:5.0
WORKDIR /code/DiffSharp
COPY . /code/DiffSharp
RUN dotnet build

