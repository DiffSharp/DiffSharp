``` ini

BenchmarkDotNet=v0.12.1, OS=Windows 10.0.17134.1667 (1803/April2018Update/Redstone4)
Intel Xeon CPU E5-1620 0 3.60GHz, 1 CPU, 8 logical and 4 physical cores
.NET Core SDK=3.1.401
  [Host]   : .NET Core 3.1.7 (CoreCLR 4.700.20.36602, CoreFX 4.700.20.37001), X64 RyuJIT DEBUG
  ShortRun : .NET Core 3.1.7 (CoreCLR 4.700.20.36602, CoreFX 4.700.20.37001), X64 RyuJIT

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
|    Method |         Mean |       Error |      StdDev |
|---------- |-------------:|------------:|------------:|
|      Add1 | 234,449.1 μs | 47,752.1 μs | 2,617.46 μs |
|      Add8 |  30,927.1 μs |  4,209.8 μs |   230.75 μs |
|     Add64 |   4,442.7 μs |    762.0 μs |    41.77 μs |
|    Add512 |   1,077.1 μs |    127.5 μs |     6.99 μs |
|   Add4096 |     685.2 μs |    284.6 μs |    15.60 μs |
|  Add32768 |     710.5 μs |    228.6 μs |    12.53 μs |
| Add262144 |     756.5 μs |    295.7 μs |    16.21 μs |
