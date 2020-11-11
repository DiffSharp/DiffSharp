namespace DiffSharp.Benchmarks

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Running
open BenchmarkDotNet.Order
open Python
open Python.Runtime

[<AutoOpen>]
module PythonHelpers =
    //#r "nuget: pythonnet_netstandard_py38_win"
    open System
    open Python.Runtime
    let execPython(code) = 
        // your mileage may differ
        if Environment.GetEnvironmentVariable("COMPUTERNAME") = "MSRC-3617253" then
            Environment.SetEnvironmentVariable("PYTHONHOME", @"C:\ProgramData\Anaconda3\", EnvironmentVariableTarget.User)
        if Environment.GetEnvironmentVariable("PYTHONHOME") = null then failwith "expect PYTHONHOME to be set"
        use gil = Py.GIL()
        use scope = Py.CreateScope()
        //scope.Exec("import torch")
        scope.Exec(code) |> ignore
//    execPython("""
//for x in range(5):
//    torch.tensor(range(5))
//""")

type BasicTensorTestMatrix() = 

    member val public workloadSize = pown 2 18 

#if TINY
    [<Params (2048)>] 
#else
    [<Params (1, 16, 2048, 65536)>] 
#endif
    member val public tensorSize = 0 with get, set

#if TINY
    [<Params ("float32")>] 
#else
    [<Params ("int32", "float32", "float64")>] 
#endif
    member val public dtypeName = "" with get, set

#if TINY
    [<Params ("cpu")>] 
#else
    [<Params ("cpu", "cuda")>] 
#endif
    member val public deviceName = "" with get, set


    member perf.numIterations = perf.workloadSize/perf.tensorSize
    member perf.caseId = sprintf "tensorSize=%d,dtypeName=%s,deviceName=%s" perf.tensorSize perf.dtypeName perf.deviceName
