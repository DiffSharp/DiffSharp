//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under the LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//


open System.Diagnostics
open DiffSharp.AD.Float64

//let duration n f =
//    let before = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
//    for i in 1..n do
//        f() |> ignore
//    let after = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
//    f(), (float (after - before)) / (float n)

let duration n f =
    let s = new System.Diagnostics.Stopwatch()
    s.Start() |> ignore
    for i in 1..n do
        f() |> ignore
    s.Stop() |> ignore
    let dur = s.ElapsedTicks
    f(), (float dur) / (float n)

//let printArray (s:System.IO.StreamWriter) (o:obj[]) =
//    for a in o do
//        match a with
//        | :? (float[]) as f -> s.WriteLine((vector f).ToString())
//        | :? (float[,]) as f -> s.WriteLine((Matrix.ofArray2D f).ToString())
//        | _ -> s.WriteLine(a.ToString())

let printb i t name =
    printfn "Running benchmark %2i / %2i %s" i  t name

type options = {
    repetitions : int;
    vectorSize : int;
    fileName : string;
    help : bool;
    changed : bool;
    }

let minRepetitions = 1000
let minVectorSize = 1

let dateTimeString (d:System.DateTime) =
    sprintf "%s%s%s%s%s%s" (d.Year.ToString()) (d.Month.ToString("D2")) (d.Day.ToString("D2")) (d.Hour.ToString("D2")) (d.Minute.ToString("D2")) (d.Second.ToString("D2"))

let defaultOptions = {
    repetitions = 10000; // > 100000 seems to work fine
    vectorSize = 10;
    fileName = sprintf "DiffSharpBenchmark%s.txt" (dateTimeString System.DateTime.Now)
    help = false;
    changed = false;
    }

let rec parseArgsRec args optionsSoFar =
    match args with
    | [] -> optionsSoFar
    | "/h"::_ | "-h"::_ | "--help"::_ | "/?"::_ | "-?"::_ -> {optionsSoFar with help = true}
    | "/f"::xs | "-f"::xs ->
        match xs with
        | f::xss -> 
            parseArgsRec xss {optionsSoFar with fileName = f; changed = true}
        | _ ->
            eprintfn "Option -f needs to be followed by a file name."
            parseArgsRec xs optionsSoFar
    | "/r"::xs | "-r"::xs ->
        match xs with
        | r::xss ->
            let couldparse, reps = System.Int32.TryParse r
            if couldparse then
                if reps < minRepetitions then
                    eprintfn "Given value for -r was too small, using the minimum: %i." minRepetitions
                    parseArgsRec xss {optionsSoFar with repetitions = minRepetitions; changed = true}
                else
                    parseArgsRec xss {optionsSoFar with repetitions = reps; changed = true}
            else
                eprintfn "Option -r was followed by an invalid value."
                parseArgsRec xs optionsSoFar
        | _ ->
            eprintfn "Option -r needs to be followed by a number."
            parseArgsRec xs optionsSoFar
    | "/vsize"::xs | "-vsize"::xs ->
        match xs with
        | s::xss ->
            let couldparse, size = System.Int32.TryParse s
            if couldparse then
                if size < minVectorSize then
                    eprintfn "Given value for -vsize was too small, using the minimum: %i." minVectorSize
                    parseArgsRec xss {optionsSoFar with vectorSize = minVectorSize; changed = true}
                else
                    parseArgsRec xss {optionsSoFar with vectorSize = size; changed = true}
            else
                eprintfn "Option -vsize was followed by an invalid value."
                parseArgsRec xs optionsSoFar
        | _ ->
            eprintfn "Option -vsize needs to be followed by a number."
            parseArgsRec xs optionsSoFar
    | x::xs ->
        eprintfn "Option \"%s\" is unrecognized." x
        parseArgsRec xs optionsSoFar

let parseArgs args =

    parseArgsRec args defaultOptions


[<EntryPoint>]
let main argv = 

    let benchmarkver = "1.0.8"

    printfn "DiffSharp Benchmarks"

    printfn "Copyright (c) 2014, 2015, National University of Ireland Maynooth."
    printfn "Written by: Atilim Gunes Baydin, Barak A. Pearlmutter\n"

    let ops = parseArgs (List.ofArray argv)

    if ops.help then
        printfn "Runs a series of benchmarks testing the operations in the DiffSharp library.\n"
        printfn "dsbench [-r repetitions] [-vsize size] [-f filename]\n"
        printfn "  -r repetitions  Specifies the number of repetitions."
        printfn "                  Higher values give more accurate results, through averaging."
        printfn "                  Default: %i" defaultOptions.repetitions
        printfn "                  Minimum:  %i" minRepetitions
        printfn "  -vsize size     Specifies the size of vector arguments for multivariate functions."
        printfn "                  Default: %i" defaultOptions.vectorSize
        printfn "                  Minimum:  %i" minVectorSize
        printfn "  -f filename     Specifies the name of the output file."
        printfn "                  If the file exists, it will be overwritten."
        printfn "                  Default: DiffSharpBenchmark + current time + .txt"
        0 // return an integer exit code
    else
        printfn "Use option -h for help on usage.\n"
    
        if not ops.changed then printfn "Using default options.\n"

        let n = ops.repetitions
        let nsymbolic = n / 1000
        let noriginal = n * 10
        let fileName = ops.fileName

        printfn "Repetitions: %A" n
        printfn "Vector size: %A" ops.vectorSize
        printfn "Output file name: %s\n" fileName

        printfn "Benchmarking module version: %s" benchmarkver
        let diffsharpver = System.Reflection.AssemblyName.GetAssemblyName("DiffSharp.dll").Version.ToString()
        printfn "DiffSharp library version: %s\n" diffsharpver

        let os = System.Environment.OSVersion.ToString()
        printfn "OS: %s" os

        let clr = System.Environment.Version.ToString()
        printfn ".NET CLR version: %s" clr

        let cpu =
            try
                let mutable cpu = ""
                let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM Win32_Processor")
                for mo in mos.Get() do
                    cpu <- mo.["name"].ToString()
                cpu
            with
                | _ -> "Unknown"
        printfn "CPU: %s" cpu

        let ram =
            try
                let mutable ram = ""
                let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM CIM_OperatingSystem")
                for mo in mos.Get() do
                    ram <- mo.["TotalVisibleMemorySize"].ToString() + " bytes"
                ram
            with
                | _ -> "Unknown"
        printfn "RAM: %s\n" ram

        printfn "Press any key to start benchmarking..."
        System.Console.ReadKey(true) |> ignore

        let started = System.DateTime.Now
        printfn "\nBenchmarking started: %A" started

        let rnd = System.Random()

        let x = rnd.NextDouble()
        let xD = DiffSharp.AD.Float64.D x
        let maprepeat = 100
        let fss (x:float) =
            let mutable v = x
            for i = 0 to maprepeat do
                v <- 4. * v * (1. - v)
            v
        let fssD (x:D) =
            let mutable v = x
            for i = 0 to maprepeat do
                v <- 4. * v * (1. - v)
            v


        let xv = Array.init ops.vectorSize (fun _ -> rnd.NextDouble())
        let xvD = DV xv

        let vv = Array.init ops.vectorSize (fun _ -> rnd.NextDouble())
        let vvD = DV vv

        let zv = Array.init 3 (fun _ -> rnd.NextDouble())
        let zvD = DV zv

        let fvs (x:float[]) =
            x |> Array.sumBy (fun v -> v * log (v / 2.))
        let fvsD (x:DV) =
            x * (log (x / 2.))

        let fvv (x:float[]) =
            [|x |> Array.sumBy (fun v -> v * log (v / 2.))
              x |> Array.sumBy (fun v -> exp (sin v))
              x |> Array.sumBy (fun v -> exp (cos v))|]
        let fvvD (x:DV) =
            toDV [x * log (x / 2.); exp (sin x) |> DV.sum; exp (cos x) |> DV.sum]

        printb 1 29 "original functions"
        let res_fss,      dur_fss =      duration noriginal (fun () -> fss x)
        let res_fssD,     dur_fssD =     duration noriginal (fun () -> fssD xD)
        let res_fvs,      dur_fvs =      duration noriginal (fun () -> fvs xv)
        let res_fvsD,     dur_fvsD =     duration noriginal (fun () -> fvsD xvD)
        let res_fvv,      dur_fvv =      duration noriginal (fun () -> fvv xv)
        let res_fvvD,     dur_fvvD =     duration noriginal (fun () -> fvvD xvD)

        printb 2 29 "diff"
        let res_diff_AD,  dur_diff_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diff fssD xD)
        let res_diff_N,   dur_diff_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff fss x)

        printb 3 29 "diff2"
        let res_diff2_AD, dur_diff2_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diff2 fssD xD)
        let res_diff2_N,  dur_diff2_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff2 fss x)

        printb 4 29 "diffn"
        let res_diffn_AD, dur_diffn_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diffn 2 fssD xD)
        let res_diffn_N,  dur_diffn_N =  0., 0.

        printb 5 29 "grad"
        let res_grad_AD,  dur_grad_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.grad fvsD xvD)
        let res_grad_N,   dur_grad_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.grad fvs xv)
        
        printb 6 29 "gradv"
        let res_gradv_AD, dur_gradv_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradv fvsD xvD vvD)
        let res_gradv_N,  dur_gradv_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradv fvs xv vv)

        printb 7 29 "hessian"
        let res_hessian_AD,  dur_hessian_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.hessian fvsD xvD)
        let res_hessian_N,   dur_hessian_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessian fvs xv)
        
        printb 8 29 "hessianv"
        let res_hessianv_AD, dur_hessianv_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.hessianv fvsD xvD vvD)
        let res_hessianv_N,  dur_hessianv_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessianv fvs xv vv)

        printb 9 29 "gradhessian"
        let res_gradhessian_AD,  dur_gradhessian_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessian fvsD xvD)
        let res_gradhessian_N,   dur_gradhessian_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessian fvs xv)
        
        printb 10 29 "gradhessianv"
        let res_gradhessianv_AD, dur_gradhessianv_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessianv fvsD xvD vvD)
        let res_gradhessianv_N,  dur_gradhessianv_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessianv fvs xv vv)

        printb 11 29 "laplacian"
        let res_laplacian_AD,  dur_laplacian_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.laplacian fvsD xvD)
        let res_laplacian_N,   dur_laplacian_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.laplacian fvs xv)

        printb 12 29 "jacobian"
        let res_jacobian_AD,  dur_jacobian_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobian fvvD xvD)
        let res_jacobian_N,   dur_jacobian_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobian fvv xv)

        printb 13 29 "jacobianv"
        let res_jacobianv_AD,  dur_jacobianv_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianv fvvD xvD vvD)
        let res_jacobianv_N,   dur_jacobianv_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianv fvv xv vv)
        
        printb 14 29 "jacobianT"
        let res_jacobianT_AD,  dur_jacobianT_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianT fvvD xvD)
        let res_jacobianT_N,   dur_jacobianT_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianT fvv xv)

        printb 15 29 "jacobianTv"
        let res_jacobianTv_AD,  dur_jacobianTv_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianTv fvvD xvD zvD)
        let res_jacobianTv_N,   dur_jacobianTv_N =   0., 0.


        //
        //
        //

        printb 16 29 "diff'"
        let res_diff'_AD,  dur_diff'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diff' fssD xD)
        let res_diff'_N,   dur_diff'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff' fss x)

        printb 17 29 "diff2'"
        let res_diff2'_AD, dur_diff2'_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diff2' fssD xD)
        let res_diff2'_N,  dur_diff2'_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff2' fss x)

        printb 18 29 "diffn'"
        let res_diffn'_AD, dur_diffn'_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.diffn' 2 fssD xD)
        let res_diffn'_N,  dur_diffn'_N =  0., 0.

        printb 19 29 "grad'"
        let res_grad'_AD,  dur_grad'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.grad' fvsD xvD)
        let res_grad'_N,   dur_grad'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.grad' fvs xv)
        
        printb 20 29 "gradv'"
        let res_gradv'_AD, dur_gradv'_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradv' fvsD xvD vvD)
        let res_gradv'_N,  dur_gradv'_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradv' fvs xv vv)

        printb 21 29 "hessian'"
        let res_hessian'_AD,  dur_hessian'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.hessian' fvsD xvD)
        let res_hessian'_N,   dur_hessian'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessian' fvs xv)
        
        printb 22 29 "hessianv'"
        let res_hessianv'_AD, dur_hessianv'_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.hessianv' fvsD xvD vvD)
        let res_hessianv'_N,  dur_hessianv'_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessianv' fvs xv vv)

        printb 23 29 "gradhessian'"
        let res_gradhessian'_AD,  dur_gradhessian'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessian' fvsD xvD)
        let res_gradhessian'_N,   dur_gradhessian'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessian' fvs xv)
        
        printb 24 29 "gradhessianv'"
        let res_gradhessianv'_AD, dur_gradhessianv'_AD = duration n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessianv' fvsD xvD vvD)
        let res_gradhessianv'_N,  dur_gradhessianv'_N =  duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessianv' fvs xv vv)

        printb 25 29 "laplacian'"
        let res_laplacian'_AD,  dur_laplacian'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.laplacian' fvsD xvD)
        let res_laplacian'_N,   dur_laplacian'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.laplacian' fvs xv)

        printb 26 29 "jacobian'"
        let res_jacobian'_AD,  dur_jacobian'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobian' fvvD xvD)
        let res_jacobian'_N,   dur_jacobian'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobian' fvv xv)

        printb 27 29 "jacobianv'"
        let res_jacobianv'_AD,  dur_jacobianv'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianv' fvvD xvD vvD)
        let res_jacobianv'_N,   dur_jacobianv'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianv' fvv xv vv)
        
        printb 28 29 "jacobianT'"
        let res_jacobianT'_AD,  dur_jacobianT'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianT' fvvD xvD)
        let res_jacobianT'_N,   dur_jacobianT'_N =   duration n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianT' fvv xv)

        printb 29 29 "jacobianTv'"
        let res_jacobianTv'_AD,  dur_jacobianTv'_AD =  duration n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianTv' fvvD xvD zvD)
        let res_jacobianTv'_N,   dur_jacobianTv'_N =   0., 0.

        let finished = System.DateTime.Now
        let duration = finished - started
        printfn "Benchmarking finished: %A\n" finished
        printfn "Total duration: %A\n" duration

        let row_originals  = toDV [dur_fss;      dur_fss;       dur_fss;       dur_fvs;      dur_fvs;       dur_fvs;         dur_fvs;          dur_fvs;             dur_fvs;              dur_fvs;           dur_fvv;          dur_fvv;           dur_fvv;           dur_fvv]
        let row_originalsD = toDV [dur_fssD;     dur_fssD;      dur_fssD;      dur_fvsD;     dur_fvsD;      dur_fvsD;        dur_fvsD;         dur_fvsD;            dur_fvsD;             dur_fvsD;          dur_fvvD;         dur_fvvD;          dur_fvvD;          dur_fvvD]
        let row_AD         = toDV [dur_diff_AD;  dur_diff2_AD;  dur_diffn_AD;  dur_grad_AD;  dur_gradv_AD;  dur_hessian_AD;  dur_hessianv_AD;  dur_gradhessian_AD;  dur_gradhessianv_AD;  dur_laplacian_AD;  dur_jacobian_AD;  dur_jacobianv_AD;  dur_jacobianT_AD;  dur_jacobianTv_AD]
        let row_N          = toDV [dur_diff_N;   dur_diff2_N;   dur_diffn_N;   dur_grad_N;   dur_gradv_N;   dur_hessian_N;   dur_hessianv_N;   dur_gradhessian_N;   dur_gradhessianv_N;   dur_laplacian_N;   dur_jacobian_N;   dur_jacobianv_N;   dur_jacobianT_N;   dur_jacobianTv_N]
        let row'_AD        = toDV [dur_diff'_AD; dur_diff2'_AD; dur_diffn'_AD; dur_grad'_AD; dur_gradv'_AD; dur_hessian'_AD; dur_hessianv'_AD; dur_gradhessian'_AD; dur_gradhessianv'_AD; dur_laplacian'_AD; dur_jacobian'_AD; dur_jacobianv'_AD; dur_jacobianT'_AD; dur_jacobianTv'_AD]
        let row'_N         = toDV [dur_diff'_N;  dur_diff2'_N;  dur_diffn'_N;  dur_grad'_N;  dur_gradv'_N;  dur_hessian'_N;  dur_hessianv'_N;  dur_gradhessian'_N;  dur_gradhessianv'_N;  dur_laplacian'_N;  dur_jacobian'_N;  dur_jacobianv'_N;  dur_jacobianT'_N;  dur_jacobianTv'_N]


        let bench = DM.ofRows [row_AD ./ row_originalsD
                               row_N  ./ row_originals]

        let bench' = DM.ofRows [row'_AD ./ row_originalsD
                                row'_N  ./ row_originals]

        let score:float = convert ((DV.sum row_AD)
                                 + (DV.sum row_N)
                                 + (DV.sum row'_AD)
                                 + (DV.sum row'_N))
    
        let score = score / (float System.TimeSpan.TicksPerSecond)
        let score = 1. / score
        let score = int (score * 100000.)

        printfn "Benchmark score: %A\n" score

        printfn "Writing results to file: %s" fileName

        let stream = new System.IO.StreamWriter(fileName, false)
        stream.WriteLine("DiffSharp Benchmarks")
        stream.WriteLine("Copyright (c) 2014, 2015, National University of Ireland Maynooth.")
        stream.WriteLine("Written by: Atilim Gunes Baydin, Barak A. Pearlmutter\r\n")
        stream.WriteLine(sprintf "Benchmarking module version: %s" benchmarkver)
        stream.WriteLine(sprintf "DiffSharp library version: %s\r\n" diffsharpver)
        stream.WriteLine(sprintf "OS: %s" os)
        stream.WriteLine(sprintf ".NET CLR version: %s" clr)
        stream.WriteLine(sprintf "CPU: %s" cpu)
        stream.WriteLine(sprintf "RAM: %s\r\n" ram)
        stream.WriteLine(sprintf "Repetitions: %A\r\n" n)
        stream.WriteLine(sprintf "Benchmarking started: %A" started)
        stream.WriteLine(sprintf "Benchmarking finished: %A" finished)
        stream.WriteLine(sprintf "Total duration: %A\r\n" duration)
        stream.WriteLine(sprintf "Benchmark score: %A\r\n" score)
    
        stream.WriteLine("Benchmark matrix 1\r\n")
        stream.WriteLine("Column labels: {diff, diff2, diffn, grad, gradv, hessian, hessianv, gradhessian, gradhessianv, laplacian, jacobian, jacobianv, jacobianT, jacobianTv}")
        stream.WriteLine("Row labels: {DiffSharp.AD, Diffsharp.Numerical}")
        stream.WriteLine(sprintf "Values: %s" (bench.ToMathematicaString()))

        stream.WriteLine("\n\nBenchmark matrix 2\r\n")
        stream.WriteLine("Column labels: {diff', diff2', diffn', grad', gradv', hessian', hessianv', gradhessian', gradhessianv', laplacian', jacobian', jacobianv', jacobianT', jacobianTv'}")
        stream.WriteLine("Row labels: {DiffSharp.AD, Diffsharp.Numerical}")
        stream.WriteLine(sprintf "Values: %s" (bench'.ToMathematicaString()))

        stream.Flush()
        stream.Close()
        

        0