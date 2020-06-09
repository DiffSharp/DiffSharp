---
title: Getting started
category: tutorial
menu_order: 1
---

# How to start in 60 seconds

1. Make sure you've installed .Net Core version defined in [global.json](global.json)
2. Run `dotnet tool restore` to install all developer tools required to build the project
3. Run `dotnet fake build` to build default target of [build script](build.fsx)
4. To run tests use `dotnet fake build -t Test`
5. To build documentation use `dotnet fake build -t Docs`
