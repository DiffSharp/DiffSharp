FROM mcr.microsoft.com/dotnet/sdk:6.0
WORKDIR /code/DiffSharp
COPY . /code/DiffSharp
RUN dotnet build
RUN dotnet test
