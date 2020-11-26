// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Benchmarks.Python

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs
open DiffSharp.Benchmarks

open System
open Python.Runtime

[<AutoOpen>]
module PythonHelpers =
    // take the lock
    let gil = Py.GIL()
    let scope = Py.CreateScope()
    // your mileage may differ
    if Environment.GetEnvironmentVariable("COMPUTERNAME") = "MSRC-3617253" then
        Environment.SetEnvironmentVariable("PYTHONHOME", @"C:\ProgramData\Anaconda3\", EnvironmentVariableTarget.User)
    if Environment.GetEnvironmentVariable("PYTHONHOME") = null then failwith "expect PYTHONHOME to be set"
    let _prepPython = scope.Exec("import torch")
    
    let execPython(code) = 
        scope.Exec(code) |> ignore

//[<ShortRunJob>]
[<MarkdownExporterAttribute.GitHub; AsciiDocExporter; HtmlExporter; CsvExporter; RPlotExporter>]
[<GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)>]
[<CategoriesColumn; BaselineColumn>]
type BasicTensorOps() = 

    inherit BasicTensorTestMatrix()

    // The tests here must match the ones above
    [<Benchmark; BenchmarkCategory("fromCpuData")>]
    member perf.fromCpuData_PyTorch() = 
        let n = perf.numIterations(2)
        execPython(sprintf """
for x in range(%d):
    torch.tensor(range(%d), dtype=torch.%s, device="%s")
""" n perf.tensorSize perf.dtypeName perf.deviceName )

#if !TINY
    [<Benchmark; BenchmarkCategory("zeros")>]
    member perf.zeros_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
res = torch.tensor(1)
for x in range(%d):
    res = torch.zeros(%d, dtype=torch.%s, device="%s")
"""  n perf.tensorSize perf.dtypeName perf.deviceName )

    [<Benchmark; BenchmarkCategory("ones")>]
    member perf.ones_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
import torch
res = torch.tensor(1)
for x in range(%d):
    res = torch.ones(%d, dtype=torch.%s, device="%s")
"""  n perf.tensorSize perf.dtypeName perf.deviceName )


    [<Benchmark; BenchmarkCategory("rand")>]
    member perf.rand_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
import torch
res = torch.tensor(1)
for x in range(%d):
    res = torch.rand(%d, dtype=torch.%s, device="%s")
"""  n perf.tensorSize perf.dtypeName perf.deviceName )


    [<Benchmark; BenchmarkCategory("addition")>]
    member perf.addition_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
t = torch.tensor(range(%d), dtype=torch.%s, device="%s")
res = t
for x in range(%d):
    res = t + t
""" perf.tensorSize perf.dtypeName perf.deviceName n )

    [<Benchmark; BenchmarkCategory("addInPlace")>]
    member perf.addInPlace_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
import torch
t = torch.tensor(range(%d), dtype=torch.%s, device="%s")
res = t
for x in range(%d):
    res = t.add_(t)
"""  perf.tensorSize perf.dtypeName perf.deviceName n )


    [<Benchmark; BenchmarkCategory("addWithAlpha")>]
    member perf.addWithAlpha_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
import torch
t = torch.tensor(range(%d), dtype=torch.%s, device="%s")
res = t
for x in range(%d):
    res = t.add(t, alpha=3)
"""  perf.tensorSize perf.dtypeName perf.deviceName n )

    [<Benchmark; BenchmarkCategory("addScalar")>]
    member perf.addScalar_PyTorch() = 
        let n = perf.numIterations(10)
        execPython(sprintf """
import torch
t = torch.tensor(range(%d), dtype=torch.%s, device="%s")
res = t
for x in range(%d):
    res = t + 1
"""  perf.tensorSize perf.dtypeName perf.deviceName n )


#endif

