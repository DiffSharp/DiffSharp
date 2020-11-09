// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

open System
open DiffSharp
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open DiffSharp.Benchmarks.BasicTensorOps

[<EntryPoint>]
let main _ = 
    let summary = BenchmarkRunner.Run<BasicTensorOps>()
    printfn "summary:"
    printfn "%s" (summary.ToString())
    //let summary2 = BenchmarkRunner.Run<Training>()
    //printfn "summary2:"
    //printfn "%s" (summary2.ToString())
    0

(*
[<ShortRunJob>]
type TensorSize_VectorAdd() = 
    let N = pown 2 18

    do dsharp.config(backend=Backend.Reference, device=Device.CPU)
    //do dsharp.config(backend=Backend.Torch, device=Device.CPU)
    //do dsharp.config(backend=Backend.Torch, device=Device.GPU)
    let n1, t1 = N, dsharp.tensor [ 1 ]
    let n8, t8 = N/8, dsharp.tensor [ 1 .. 8 ]
    let n64, t64 = N/64, dsharp.tensor [ 1 .. 64 ]
    let n512, t512 = N/512, dsharp.tensor [ 1 .. 512 ]
    let n4096, t4096 = N/4096, dsharp.tensor [ 1 .. 4096 ]
    let n32768, t32768 = N/32768, dsharp.tensor [ 1 .. 32768 ]
    let n262144, t262144 = N/262144 , dsharp.tensor [| 1.0 .. 262144.0 |]

    let mutable res = dsharp.tensor 1

    [<Benchmark>]
    member _.Add1() = for i in 1 .. n1 do res <- t1 + t1

    [<Benchmark>]
    member _.Add8() = for i in 1 .. n8 do res <- t8 + t8

    [<Benchmark>]
    member _.Add64() = for i in 1 .. n64 do res <- t64 + t64

    [<Benchmark>]
    member _.Add512() = for i in 1 .. n512 do res <- t512 + t512

    [<Benchmark>]
    member _.Add4096() = for i in 1 .. n4096 do res <- t4096 + t4096

    [<Benchmark>]
    member _.Add32768() = for i in 1 .. n32768 do res <- t32768 + t32768

    [<Benchmark>]
    member _.Add262144() = for i in 1 .. n262144 do res <- t262144 + t262144


let summary = BenchmarkRunner.Run<TensorSize_VectorAdd>()

module Analysis =

    let N = pown 2 18

    // Fit a pair of points to a linear model 
    //    t = c1 * N + c2 * (N/n)
    //    c1 = cost per tensor element
    //    c2 = cost per tensor and/or tensor addition operation
    let fitp (n1, r1) (n2, r2) = 
        let c2 = (r1 - r2) / (float (N / n1) - float (N / n2))
        let c1 = (r1 - c2 * float (N / n1)) / float N
        c1, c2

    let fit data = 
        data
        |> List.mapi (fun i t -> pown 2 (i * 3), t) 
        |> List.pairwise 
        |> List.map (fun (p1,p2) -> fitp p1 p2)
        |> List.unzip 
        // average but discard last which seems to have major cache effects
        //|> (fun (a,b) -> List.average a.[0..a.Length-2], List.average b.[0..b.Length-2])

(*

1. Baseline 07/09/2020

    |    Method |           Mean |        Error |       StdDev |
    |---------- |---------------:|-------------:|-------------:|
    |      Add1 | 2_084_633.9 μs | 41_485.88 μs | 59_497.80 μs |
    |      Add8 |   256_473.3 μs |  4_598.87 μs |  5_111.64 μs |
    |     Add64 |    32_725.8 μs |    460.31 μs |    408.06 μs |
    |    Add512 |     4_911.4 μs |     85.78 μs |     80.24 μs |
    |   Add4096 |     1_599.1 μs |     24.38 μs |     45.19 μs |
    |  Add32768 |     1_008.0 μs |     19.75 μs |     23.52 μs |
    | Add262144 |       388.7 μs |      4.98 μs |      4.16 μs |

*)

    let fits1 = fit [ 2_084_633.0; 256_473.0; 32_725.0; 4_911.0; 1_599.0; 1_008.0; 388.7 ] 

(*

2. With 4 dummy copies of int32[] shape in each RawTensor

|    Method |           Mean |        Error |       StdDev |         Median |
|---------- |---------------:|-------------:|-------------:|---------------:|
|      Add1 | 2_195_873.9 us | 19_222.59 us | 17_040.32 us | 2_196_215.4 us |
|      Add8 |   276_061.1 us |  4_989.21 us |  8_056.64 us |   274_654.9 us |
|     Add64 |    35_767.1 us |    599.06 us |    560.36 us |    35_627.3 us |
|    Add512 |     5_101.8 us |     96.52 us |    107.28 us |     5_086.1 us |
|   Add4096 |     1_619.1 us |     31.89 us |     50.58 us |     1_626.2 us |
|  Add32768 |     1_018.9 us |     19.38 us |     41.72 us |       997.0 us |
| Add262144 |       388.9 us |      4.97 us |      3.88 us |       390.2 us |
*)
    let fits2 = fit [ 2_195_873.0; 276_061.0; 35_767.0; 5_101.0; 1_619.0; 1_018.0; 388.7 ]

(*

3. 1 + Removing checks in TorchTensor creation and removing 4 fields from RawTensor

|    Method |           Mean |        Error |       StdDev |         Median |
|---------- |---------------:|-------------:|-------------:|---------------:|
|      Add1 | 1_731_513.6 us | 33_259.25 us | 38_301.40 us | 1_718_709.8 us |
|      Add8 |   233_161.8 us |  4_590.32 us |  6_583.30 us |   230_817.9 us |
|     Add64 |    28_019.9 us |    365.76 us |    324.24 us |    27_958.7 us |
|    Add512 |     5_770.2 us |    187.73 us |    553.52 us |     5_931.7 us |
|   Add4096 |     1_532.4 us |     25.96 us |     24.28 us |     1_531.8 us |
|  Add32768 |     1_026.2 us |     25.73 us |     75.47 us |     1_012.8 us |
| Add262144 |       372.6 us |      7.42 us |      7.29 us |       371.9 us |
*)
    let fits3 = fit [ 1_731_513.0; 233_161.0; 28_019.0; 5_770.0; 1_532.0; 1_026.0; 372.7 ] 

(*

4. 3 + removing "device" fields from TorchTensor + running on Torch CPU

|    Method |           Mean |           Error |       StdDev |
|---------- |---------------:|----------------:|-------------:|
|      Add1 | 1_791_513.0 us | 1_057_817.29 us | 57_982.54 us |
|      Add8 |   225_713.3 us |   110_940.05 us |  6_081.00 us |
|     Add64 |    28_231.9 us |     5_307.85 us |    290.94 us |
|    Add512 |     4_289.1 us |     2_075.17 us |    113.75 us |
|   Add4096 |     1_462.1 us |       327.50 us |     17.95 us |
|  Add32768 |     1_026.8 us |       202.28 us |     11.09 us |
| Add262144 |       389.5 us |        65.88 us |      3.61 us |
*)

    let fits4 = fit [ 1_791_513.0; 225_713.0; 28_231.0; 4_289.0; 1_462.; 1_026.; 389. ] 

(*

5. 3 + removing "device" fields from TorchTensor + running on GPU

|    Method |             Mean |            Error |         StdDev |
|---------- |-----------------:|-----------------:|---------------:|
|      Add1 | 19_576_903.47 us | 11_596_251.89 us | 635_629.696 us |
|      Add8 |  2_201_603.13 us |     39_920.38 us |   2_188.171 us |
|     Add64 |    420_980.60 us |     30_840.46 us |   1_690.469 us |
|    Add512 |     34_422.14 us |        333.76 us |      18.294 us |
|   Add4096 |      5_906.75 us |      1_541.00 us |      84.468 us |
|  Add32768 |        533.04 us |         43.67 us |       2.394 us |
| Add262144 |         79.60 us |        379.62 us |      20.808 us |

*)

    let fits5 = fit [ 19_576_903.0; 2_201_603.0; 420_980.0; 34_422.0; 5_906.0; 533.0; 79.6 ] 

(*

6. 3 + running on Reference backend

|    Method |         Mean |         Error |       StdDev |
|---------- |-------------:|--------------:|-------------:|
|      Add1 | 242_813.9 us | 192_225.46 us | 10_536.53 us |
|      Add8 |  30_931.0 us |  15_950.54 us |    874.30 us |
|     Add64 |   4_596.7 us |   1_736.71 us |     95.20 us |
|    Add512 |   1_080.2 us |     219.47 us |     12.03 us |
|   Add4096 |     676.8 us |      60.02 us |      3.29 us |
|  Add32768 |     737.9 us |   1_333.39 us |     73.09 us |
| Add262144 |     770.2 us |     203.48 us |     11.15 us |
*)

    let fits6 = fit [ 242_813.0; 30_931.0; 4_596.0; 1_080.0; 676.0; 737.0; 770.2 ] 

*)
