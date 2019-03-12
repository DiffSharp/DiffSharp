(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Download
========

## Windows

You can install the library via its [NuGet package](https://www.nuget.org/packages/diffsharp). 

If you don't use NuGet, you can download the binaries of the latest release <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

For using DiffSharp, your project should target .NET Framework 4.6 or higher before installing the NuGet package. 

Starting with version 0.7, DiffSharp only supports the 64 bit platform. In the build configuration of your project, you should set "x64" as the platform target (don't forget to do this for all build configurations). 
If you need to use 32-bit, please adjust the source.

If you are using F# interactive, you should run it in 64 bit mode. In Visual Studio, you can do this by selecting "Tools - Options - F# Tools - F# Interactive" and setting "64 bit F# Interactive" to "true" and restarting the IDE.

<div class="row">
    <div class="span1"></div>
    <div class="span7">
    <div class="well well-small" id="nuget">
        To install <a href="https://www.nuget.org/packages/diffsharp">DiffSharp on NuGet</a>, run the following in the <a href="https://docs.nuget.org/docs/start-here/using-the-package-manager-console">Package Manager Console</a>:
        <pre>PM> Install-Package DiffSharp</pre>
    </div>
    </div>
    <div class="span1"></div>
</div>

## Linux

Please make sure you have the latest **libopenblas-dev** package installed for OpenBLAS.

You should have a working .NET runtime on your system. [Mono](https://www.mono-project.com/) has been the standard choice for Linux, but the community is in the process of moving to [.NET Core](https://dotnet.github.io/), a new cross-platform implementation of the framework. Please refer to [fsharp.org](https://fsharp.org/) for the latest instructions.

If you have a .NET setup where you can use NuGet, once you have the file _libopenblas.so_ in the library search path (e.g. in /usr/lib), you can use the same NuGet package described above.

Alternatively, you can download the Linux-specific pack of binaries of the latest release, which also includes a compatible version of _libopenblas.so_, <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

You can check out [Ionide](https://ionide.io/), a lightweight editor for F# development on Linux.

## FAQ

##### "I get a _System.EntryPointNotFoundException_ when running my code."

This is because you have an old version of OpenBLAS on your system. DiffSharp uses of _?omatcopy_ and _?imatcopy_ extensions for BLAS [introduced by Intel MKL](https://software.intel.com/en-us/node/520863) for fast matrix transposition, which were not present in earlier versions of OpenBLAS. 

On Linux, you can compile the latest OpenBLAS using [these instructions](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide). We also distribute a compatible version of _libopenblas.so_ in the Linux-specific pack of the latest release <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

Please make sure you have the latest version of _libopenblas.so_ in the shared library search path. Also see the "Linux shared library search path" section on [this page](https://www.mono-project.com/docs/advanced/pinvoke/).

<br>

##### "When trying to use DiffSharp with an .fsx script in F# Interactive, I get 'Unable to load DLL libopenblas'. When I compile the code as a .fs code file, everything runs fine."

This is related with the general behavior of F# Interactive and how it works with native dlls. It is not specific to DiffSharp.

[This post](https://christoph.ruegg.name/blog/loading-native-dlls-in-fsharp-interactive.html) by Christoph Rüegg provides a detailed overview of how to load native libraries for scripts.

In short, you have to make sure that you have the OpenBLAS binaries (_libopenblas.dll, libgcc_s_seh-1.dll, libgfortran-3.dll, libquadmath-0.dll_ on Windows, and _libopenblas.so_ on Linux) in a location reachable by the _DiffSharp.dll_ assembly you are loading into your script (e.g., _#r "../DiffSharp.dll"_).

On Linux, make sure that you have _libopenblas.so_ in the shared library search path. Also see the "Linux shared library search path" section on [this page](https://www.mono-project.com/docs/advanced/pinvoke/).

On Windows, one way of accomplishing this is to put the _DiffSharp.dll_ and OpenBLAS binaries into the same folder with your .fsx script and call

    System.Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

at the beginning of your script.

*)