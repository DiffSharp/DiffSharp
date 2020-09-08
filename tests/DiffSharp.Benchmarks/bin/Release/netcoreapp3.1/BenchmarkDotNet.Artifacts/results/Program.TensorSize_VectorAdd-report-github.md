``` ini

BenchmarkDotNet=v0.12.1, OS=Windows 10.0.17134.1667 (1803/April2018Update/Redstone4)
Intel Xeon CPU E5-1620 0 3.60GHz, 1 CPU, 8 logical and 4 physical cores
.NET Core SDK=3.1.401
  [Host]   : .NET Core 3.1.7 (CoreCLR 4.700.20.36602, CoreFX 4.700.20.37001), X64 RyuJIT DEBUG
  ShortRun : .NET Core 3.1.7 (CoreCLR 4.700.20.36602, CoreFX 4.700.20.37001), X64 RyuJIT

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
|    Method |             Mean |            Error |         StdDev |
|---------- |-----------------:|-----------------:|---------------:|
|      Add1 | 19,576,903.47 μs | 11,596,251.89 μs | 635,629.696 μs |
|      Add8 |  2,201,603.13 μs |     39,920.38 μs |   2,188.171 μs |
|     Add64 |    420,980.60 μs |     30,840.46 μs |   1,690.469 μs |
|    Add512 |     34,422.14 μs |        333.76 μs |      18.294 μs |
|   Add4096 |      5,906.75 μs |      1,541.00 μs |      84.468 μs |
|  Add32768 |        533.04 μs |         43.67 μs |       2.394 μs |
| Add262144 |         79.60 μs |        379.62 μs |      20.808 μs |
