// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

open System.IO
open BenchmarkDotNet.Running
open DiffSharp.Benchmarks.Python

[<EntryPoint>]
let main _ = 

    let summary = BenchmarkRunner.Run<BasicTensorOps>()
    printfn "summary:"
    printfn "%s" (summary.ToString())

    let lines = 
        [ for case in summary.BenchmarksCases do
            if case.Descriptor.Categories.Length > 0 then
                let report = summary.[case]
                let tensorSize = case.Parameters.["tensorSize"] :?> int
                let dtypeName = case.Parameters.["dtypeName"] :?> string
                let deviceName = case.Parameters.["deviceName"] :?> string 
                // get the time in milliseconds
                let runtime = report.ResultStatistics.Mean / 1000000.0 |> int64
                let nm = case.Descriptor.Categories.[0]
                let key = nm + string tensorSize + dtypeName + deviceName
                sprintf "%s,%d" key runtime 
        ]

    File.WriteAllLines(Path.Combine(__SOURCE_DIRECTORY__, "results.csv"), lines)

    0
