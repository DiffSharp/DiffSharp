(*** condition: prepare ***)
#r "../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget:RestoreSources=https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#i "nuget: https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"
#endif // IPYNB

(**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=gett-ingstarted-topic1.ipynb)

Nested AD
=========

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.



*)

