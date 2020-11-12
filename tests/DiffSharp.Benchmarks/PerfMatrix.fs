namespace DiffSharp.Benchmarks

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Running
open BenchmarkDotNet.Order

type BasicTensorTestMatrix() = 

    member val public workloadSize = pown 2 18 

#if TINY
    [<Params (2048)>] 
#else
    [<Params (16, 2048, 65536)>] 
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

    member perf.numIterations(factor) = factor * perf.workloadSize / perf.tensorSize
    member perf.caseId = sprintf "tensorSize=%d,dtypeName=%s,deviceName=%s" perf.tensorSize perf.dtypeName perf.deviceName
