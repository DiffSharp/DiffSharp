// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

open System
open System.IO
open DiffSharp
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running
open BenchmarkDotNet.Characteristics
open DiffSharp.Benchmarks

[<EntryPoint>]
let main _ = 

    let summary = BenchmarkRunner.Run<BasicTensorOps>()
    printfn "summary:"
    printfn "%s" (summary.ToString())

#if UPDATEPYTHON
    let mutable contents = File.ReadAllLines(BasicTensorOps.ThisFile)
    let groups = 
        summary.BenchmarksCases 
        |> Seq.filter (fun case -> case.Descriptor.Categories.Length = 1) 
        |> Seq.filter (fun case -> summary.[case].Success) 
        |>  Seq.groupBy(fun case -> case.Descriptor.Categories |> Array.tryHead)

    // Get the Python times and write them back in to the program text
    // Python torch can't easily cohabitate with DiffSharp torch in the same process
    // due to multiple copies of native LibTorch DLLs being loaded
    for (category, cases) in groups do
       printfn "category = %A" category
       match category with 
       | None -> ()
       | Some cat -> 
           let key = "// PYTHON " + cat
           let newPythonLineText = 
               [ for case in cases do 
                    let report = summary.[case]
                    for p in case.Parameters.Items do
                        printfn " %s --> %O" p.Name p.Value
                    let tensorSize = case.Parameters.["tensorSize"] :?> int
                    let dtypeName = case.Parameters.["dtypeName"] :?> string
                    let deviceName = case.Parameters.["deviceName"] :?> string 
                    // get the time in milliseconds
                    let runtime = report.ResultStatistics.Mean / 1000000.0 |> int64
                    sprintf "if perf.tensorSize = %d && perf.dtypeName = \"%s\" && perf.deviceName = \"%s\" then Thread.Sleep(%d) el" tensorSize dtypeName deviceName runtime ]
               |> String.concat ""
               |> fun s -> "        " + s + "se failwith \"no time available\" " + key

           printfn "looking for %s...." key
           contents <- 
               contents |> Array.map (fun line ->
                   if line.Contains(key) then
                       printfn "found %s, new text = %s" key newPythonLineText
                       newPythonLineText else line)

    File.WriteAllLines(BasicTensorOps.ThisFile, contents)
#endif
    //let summary2 = BenchmarkRunner.Run<Training>()
    //printfn "summary2:"
    //printfn "%s" (summary2.ToString())
    0
