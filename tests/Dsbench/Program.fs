// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

open System.Diagnostics
open DiffSharp.AD.Float64

//let duration n f =
//    let before = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
//    for i in 1..n do
//        f() |> ignore
//    let after = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
//    f(), (float (after - before)) / (float n)

let duration nm n f () =
    let s = new System.Diagnostics.Stopwatch()
    printfn "* running %s" nm
    s.Start() |> ignore
    for i in 1..n do
        f() |> ignore
    s.Stop() |> ignore
    let dur = s.ElapsedTicks
    (float dur) / (float n)

//let printArray (s:System.IO.StreamWriter) (o:obj[]) =
//    for a in o do
//        match a with
//        | :? (float[]) as f -> s.WriteLine((vector f).ToString())
//        | :? (float[,]) as f -> s.WriteLine((Matrix.ofArray2D f).ToString())
//        | _ -> s.WriteLine(a.ToString())

let printb i t name =
    printfn "Running benchmark %2i / %2i %s" i  t name

let opArgNames =
     ["diff";
      "diff2";
      "diffn";
      "computeAdjointsV";
      "computeAdjointsM";
      "gradV";
      "gradv";
      "hessian";
      "hessianv";
      "gradhessian";
      "gradhessianv";
      "laplacian";
      "jacobian";
      "jacobianv";
      "jacobianT";
      "jacobianTv"]

type options = {
    benchmarks : string list
    modes : string list
    opsToRun : string list
    repetitions : int
    vectorSize : int
    matrixSize : int
    fileName : string
    help : bool
    changed : bool
    }

let minRepetitions = 1
let minVectorSize = 1
let minMatrixSize = 1

let dateTimeString (d:System.DateTime) =
    sprintf "%s%s%s%s%s%s" (d.Year.ToString()) (d.Month.ToString("D2")) (d.Day.ToString("D2")) (d.Hour.ToString("D2")) (d.Minute.ToString("D2")) (d.Second.ToString("D2"))

let defaultOptions = {
    benchmarks = []
    modes = []
    opsToRun = []
    repetitions = 10000 // > 100000 seems to work fine
    vectorSize = 10
    matrixSize = 10
    fileName = sprintf "DiffSharpBenchmark%s.txt" (dateTimeString System.DateTime.Now)
    help = false
    changed = false
    }

let rec parseArgsRec args optionsSoFar =
    match args with
    | [] -> optionsSoFar
    | "/h"::_ | "-h"::_ | "--help"::_ | "/?"::_ | "-?"::_ -> {optionsSoFar with help = true}
    | ("/b" | "-b")::xs ->
        match xs with
        | (("1" | "2") as f)::xss ->
            parseArgsRec xss {optionsSoFar with benchmarks= optionsSoFar.benchmarks @ [f]}
        | _ ->
            eprintfn "Option -b needs to be followed by a benchmark name (1, 2)."
            exit 1
    | ("/m" | "-m") ::xs ->
        match xs with
        | (("auto" | "numeric") as f)::xss ->
            parseArgsRec xss {optionsSoFar with modes = optionsSoFar.modes @ [f]}
        | _ ->
            eprintfn "Option -b needs to be followed by a mode name (auto,numeric)."
            exit 1
    | ("/o" | "-o") ::xs ->
        match xs with
        | f::xss when List.contains f opArgNames ->
            parseArgsRec xss {optionsSoFar with opsToRun = optionsSoFar.opsToRun @ [f]}
        | _ ->
            eprintfn "Option -o needs to be followed by a operation name (%s)." (String.concat "," opArgNames)
            exit 1
    | "/f"::xs | "-f"::xs ->
        match xs with
        | f::xss ->
            parseArgsRec xss {optionsSoFar with fileName = f; changed = true}
        | _ ->
            eprintfn "Option -f needs to be followed by a file name."
            exit 1
    | "/r"::xs | "-r"::xs ->
        match xs with
        | r::xss ->
            let couldparse, reps = System.Int32.TryParse r
            if couldparse then
                if reps < minRepetitions then
                    eprintfn "Given value for -r was too small, using the minimum: %i." minRepetitions
                    exit 1
                else
                    parseArgsRec xss {optionsSoFar with repetitions = reps; changed = true}
            else
                eprintfn "Option -r was followed by an invalid value."
                exit 1
        | _ ->
            eprintfn "Option -r needs to be followed by a number."
            exit 1
    | "/vsize"::xs | "-vsize"::xs ->
        match xs with
        | s::xss ->
            let couldparse, size = System.Int32.TryParse s
            if couldparse then
                if size < minVectorSize then
                    eprintfn "Given value for -vsize was too small, using the minimum: %i." minVectorSize
                    exit 1
                else
                    parseArgsRec xss {optionsSoFar with vectorSize = size; changed = true}
            else
                eprintfn "Option -vsize was followed by an invalid value."
                exit 1
        | _ ->
            eprintfn "Option -vsize needs to be followed by a number."
            exit 1
    | "/msize"::xs | "-msize"::xs ->
        match xs with
        | s::xss ->
            let couldparse, size = System.Int32.TryParse s
            if couldparse then
                if size < minMatrixSize then
                    eprintfn "Given value for -msize was too small, using the minimum: %i." minMatrixSize
                    exit 1
                else
                    parseArgsRec xss {optionsSoFar with matrixSize = size; changed = true}
            else
                eprintfn "Option -msize was followed by an invalid value."
                exit 1
        | _ ->
            eprintfn "Option -vsize needs to be followed by a number."
            exit 1
    | x::xs ->
        eprintfn "Option \"%s\" is unrecognized." x
        parseArgsRec xs optionsSoFar

let parseArgs args =

    parseArgsRec args defaultOptions


[<EntryPoint>]
let main argv =

    let benchmarkver = "1.0.9"

    printfn "DiffSharp Benchmarks"

    printfn "Copyright (c) 2014--2016, National University of Ireland Maynooth."
    printfn "Written by: Atilim Gunes Baydin, Barak A. Pearlmutter\n"

    let ops = parseArgs (List.ofArray argv)

    if ops.help then
        printfn "Runs a series of benchmarks testing the operations in the DiffSharp library.\n"
        printfn "dsbench [-r repetitions] [-vsize size] [-f filename]\n"
        printfn "  -b benchmark    Specifies the benchmark (1 or 2)."
        printfn "  -m mode         Specifies the mode (auto or numeric)."
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

        let benchmarks =
            match ops.benchmarks with
            | [] -> ["1"; "2"]
            | xss -> xss

        let modes =
            match ops.modes with
            | [] -> [ "auto"; "numeric"]
            | xss -> xss

        let opsToRun =
            match ops.opsToRun with
            | [] -> opArgNames
            | xss -> xss

        let bench1 = List.contains "1" benchmarks
        let bench2 = List.contains "2" benchmarks
        let auto = List.contains "auto" modes
        let numeric = List.contains "numeric" modes

        let n = ops.repetitions
        let nsymbolic = n / 1000
        let noriginal = n * 10
        let fileName = ops.fileName

        printfn "Repetitions: %A" n
        printfn "Vector size: %A" ops.vectorSize
        printfn "Matrix size: %A" ops.matrixSize
        printfn "Ops to run : %s" (String.concat "," ops.opsToRun)
        printfn "Modes : %s" (String.concat "," ops.modes)
        printfn "Output file name: %s\n" fileName

        printfn "Benchmarking module version: %s" benchmarkver
        let diffsharpver = typeof<DiffSharp.AD.Float64.D>.Assembly.GetName().Version.ToString()
        printfn "DiffSharp library version: %s\n" diffsharpver

        let os = System.Environment.OSVersion.ToString()
        printfn "OS: %s" os

        let clr = System.Environment.Version.ToString()
        printfn ".NET CLR version: %s" clr

        let cpu = "Unknown"
//            try
//                let mutable cpu = ""
//                let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM Win32_Processor")
//                for mo in mos.Get() do
//                    cpu <- mo.["name"].ToString()
//                cpu
//            with
//                | _ -> "Unknown"
        printfn "CPU: %s" cpu

        let ram = "Unknown"
//            try
//                let mutable ram = ""
//                let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM CIM_OperatingSystem")
//                for mo in mos.Get() do
//                    ram <- mo.["TotalVisibleMemorySize"].ToString() + " bytes"
//                ram
//            with
//                | _ -> "Unknown"
        printfn "RAM: %s\n" ram

        //printfn "Press any key to start benchmarking..."
        //System.Console.ReadKey(true) |> ignore

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

        let xm = Array2D.init ops.matrixSize ops.matrixSize (fun _ _ -> rnd.NextDouble())
        let xmD = DM xm

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

        let fms (x:float[,]) =
            x |> Array2D.map (fun v -> v * log (v / 2.))

        let fmsD (x:DM) =
            x * (log (x / 2.)) |> DM.max

        let fmm (x:float[,]) =
            [|x |> Array2D.map (fun v -> exp (sin v))
              x |> Array2D.map (fun v -> exp (cos v))|]
        let fmmD (x:DM) =
            toDV [exp (sin x) |> DM.sum; exp (cos x) |> DM.sum]

        let run_fss =      duration "eval" noriginal (fun () -> fss x)
        let run_fssD =     duration "evalD" noriginal (fun () -> fssD xD)

        let run_fvs =      duration "evalv" noriginal (fun () -> fvs xv)
        let run_fvsD =     duration "evalvD" noriginal (fun () -> fvsD xvD)
        let run_fvv =      duration "evalv" noriginal (fun () -> fvv xv)
        let run_fvvD =     duration "evalvD" noriginal (fun () -> fvvD xvD)

        let run_fms =      duration "evalm" noriginal (fun () -> fms xm)
        let run_fmsD =     duration "evalmD" noriginal (fun () -> fmsD xmD)
        let run_fmm =      duration "evalm" noriginal (fun () -> fmm xm)
        let run_fmmD =     duration "evalmD" noriginal (fun () -> fmmD xmD)

        let run_diff_AD =  duration "diff (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diff fssD xD)
        let run_diff_N =   duration "diff (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff fss x)

        let run_diff2_AD = duration "diff2 (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diff2 fssD xD)
        let run_diff2_N =  duration "diff2 (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff2 fss x)

        let run_diffn_AD = duration "diffn (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diffn 2 fssD xD)
        let run_diffn_N() =  0.

        let run_grad_AD =  duration "grad (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.grad fvsD xvD)
        let run_grad_N =   duration "grad (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.grad fvs xv)

        let run_computeAdjoints_AD =  duration "computeAdjoints (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.computeAdjoints (fvsD (DiffSharp.AD.Float64.DOps.makeReverse 100u xvD)))

        let run_computeAdjoints_AD_m =  duration "computeAdjoints (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.computeAdjoints (fmsD (DiffSharp.AD.Float64.DOps.makeReverse 100u xmD)))

        let run_gradv_AD = duration "gradv (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradv fvsD xvD vvD)
        let run_gradv_N =  duration "gradv (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradv fvs xv vv)

        let run_hessian_AD =  duration "hessian (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.hessian fvsD xvD)
        let run_hessian_N =   duration "hessian (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessian fvs xv)

        let run_hessianv_AD = duration "hessianv (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.hessianv fvsD xvD vvD)
        let run_hessianv_N =  duration "hessianv (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessianv fvs xv vv)

        let run_gradhessian_AD =  duration "gradhessian (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessian fvsD xvD)
        let run_gradhessian_N =   duration "gradhessian (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessian fvs xv)

        let run_gradhessianv_AD = duration "gradhessianv (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessianv fvsD xvD vvD)
        let run_gradhessianv_N =  duration "gradhessianv (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessianv fvs xv vv)

        let run_laplacian_AD =  duration "laplacian (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.laplacian fvsD xvD)
        let run_laplacian_N =   duration "laplacian (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.laplacian fvs xv)

        let run_jacobian_AD =  duration "jacobian (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobian fvvD xvD)
        let run_jacobian_N =   duration "jacobian (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobian fvv xv)

        let run_jacobianv_AD =  duration "jacobianv (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianv fvvD xvD vvD)
        let run_jacobianv_N =   duration "jacobianv (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianv fvv xv vv)

        let run_jacobianT_AD =  duration "jacobianT (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianT fvvD xvD)
        let run_jacobianT_N =   duration "jacobianT (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianT fvv xv)

        let run_jacobianTv_AD =  duration "jacobianTv (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianTv fvvD xvD zvD)
        let run_jacobianTv_N () =   0.


        let run_diff'_AD =  duration "diff' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diff' fssD xD)
        let run_diff'_N =   duration "diff' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff' fss x)

        let run_diff2'_AD = duration "diff2' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diff2' fssD xD)
        let run_diff2'_N =  duration "diff2' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.diff2' fss x)

        let run_diffn'_AD = duration "diffn' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.diffn' 2 fssD xD)
        let run_diffn'_N() =  0.

        let run_grad'_AD =  duration "grad' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.grad' fvsD xvD)
        let run_grad'_N =   duration "grad' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.grad' fvs xv)

        let run_gradv'_AD = duration "gradv' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradv' fvsD xvD vvD)
        let run_gradv'_N =  duration "gradv' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradv' fvs xv vv)

        let run_hessian'_AD =  duration "hessian' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.hessian' fvsD xvD)
        let run_hessian'_N =   duration "hessian' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessian' fvs xv)

        let run_hessianv'_AD = duration "hessianv' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.hessianv' fvsD xvD vvD)
        let run_hessianv'_N =  duration "hessianv' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.hessianv' fvs xv vv)

        let run_gradhessian'_AD =  duration "gradhessian' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessian' fvsD xvD)
        let run_gradhessian'_N =   duration "gradhessian' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessian' fvs xv)

        let run_gradhessianv'_AD = duration "gradhessianv' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.gradhessianv' fvsD xvD vvD)
        let run_gradhessianv'_N =  duration "gradhessianv' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.gradhessianv' fvs xv vv)

        let run_laplacian'_AD =  duration "laplacian' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.laplacian' fvsD xvD)
        let run_laplacian'_N =   duration "laplacian' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.laplacian' fvs xv)

        let run_jacobian'_AD =  duration "jacobian' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobian' fvvD xvD)
        let run_jacobian'_N =   duration "jacobian' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobian' fvv xv)

        let run_jacobianv'_AD =  duration "jacobianv' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianv' fvvD xvD vvD)
        let run_jacobianv'_N =   duration "jacobianv' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianv' fvv xv vv)

        let run_jacobianT'_AD =  duration "jacobianT' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianT' fvvD xvD)
        let run_jacobianT'_N =   duration "jacobianT' (numeric)" n (fun () -> DiffSharp.Numerical.Float64.DiffOps.jacobianT' fvv xv)

        let run_jacobianTv'_AD =  duration "jacobianTv' (auto)" n (fun () -> DiffSharp.AD.Float64.DiffOps.jacobianTv' fvvD xvD zvD)
        let run_jacobianTv'_N () =   0.

        let finished = System.DateTime.Now
        let duration = finished - started
        printfn "Benchmarking finished: %A\n" finished
        printfn "Total duration: %A\n" duration

        let fss = lazy run_fss()
        let fvs = lazy run_fvs()
        let fvv = lazy run_fvv()
        let fms = lazy run_fms()
        let fmm = lazy run_fmm()
        let fssD = lazy run_fssD()
        let fvsD = lazy run_fvsD()
        let fvvD = lazy run_fvvD()
        let fmsD = lazy run_fmsD()
        let fmmD = lazy run_fmmD()
        let k v () = v
        let K (vl: Lazy<_>) () = vl.Force()

        let opNames        = ["diff";       "diff2";       "diffn";       "computeAdjointsV";       "computeAdjointsM";       "gradV";      "gradv";       "hessian";       "hessianv";       "gradhessian";       "gradhessianv";       "laplacian";       "jacobian";       "jacobianv";       "jacobianT";       "jacobianTv"]
        let row_originals  = [K fss;        K fss;         K fss;         K fvs;                   K fms;                     K fvs;        K fvs;         K fvs;           K fvs;            K fvs;               K fvs;                K fvs;             K fvv;            K fvv;             K fvv;             K fvv]
        let row_originalsD = [K fssD;       K fssD;        K fssD;        K fvsD;                  K fmsD;                    K fvsD;       K fvsD;        K fvsD;          K fvsD;           K fvsD;              K fvsD;               K fvsD;            K fvvD;           K fvvD;            K fvvD;            K fvvD]
        let row_AD         = [run_diff_AD;  run_diff2_AD;  run_diffn_AD;  run_computeAdjoints_AD;  run_computeAdjoints_AD_m;  run_grad_AD;  run_gradv_AD;  run_hessian_AD;  run_hessianv_AD;  run_gradhessian_AD;  run_gradhessianv_AD;  run_laplacian_AD;  run_jacobian_AD;  run_jacobianv_AD;  run_jacobianT_AD;  run_jacobianTv_AD]
        let row_N          = [run_diff_N;   run_diff2_N;   run_diffn_N;   k 0.0;                   k 0.0;                     run_grad_N;   run_gradv_N;   run_hessian_N;   run_hessianv_N;   run_gradhessian_N;   run_gradhessianv_N;   run_laplacian_N;   run_jacobian_N;   run_jacobianv_N;   run_jacobianT_N;   run_jacobianTv_N]
        let row'_AD        = [run_diff'_AD; run_diff2'_AD; run_diffn'_AD; k 0.0;                   k 0.0;                     run_grad'_AD; run_gradv'_AD; run_hessian'_AD; run_hessianv'_AD; run_gradhessian'_AD; run_gradhessianv'_AD; run_laplacian'_AD; run_jacobian'_AD; run_jacobianv'_AD; run_jacobianT'_AD; run_jacobianTv'_AD]
        let row'_N         = [run_diff'_N;  run_diff2'_N;  run_diffn'_N;  k 0.0;                   k 0.0;                     run_grad'_N;  run_gradv'_N;  run_hessian'_N;  run_hessianv'_N;  run_gradhessian'_N;  run_gradhessianv'_N;  run_laplacian'_N;  run_jacobian'_N;  run_jacobianv'_N;  run_jacobianT'_N;  run_jacobianTv'_N]

        if Set.ofList opNames <> Set.ofList opArgNames then failwith "opNames <> opArgNames - fix the benchmark program please"
        let doOp op = List.contains op opsToRun
        let opNames2        = opNames |> List.filter doOp
        let row_originals  = List.zip opNames row_originals |> List.filter (fst >> doOp) |> List.map snd
        let row_originalsD = List.zip opNames row_originalsD |> List.filter (fst >> doOp) |> List.map snd
        let row_AD         = List.zip opNames row_AD |> List.filter (fst >> doOp) |> List.map snd
        let row_N          = List.zip opNames row_N |> List.filter (fst >> doOp) |> List.map snd
        let row'_AD        = List.zip opNames row'_AD |> List.filter (fst >> doOp) |> List.map snd
        let row'_N         = List.zip opNames row'_N |> List.filter (fst >> doOp) |> List.map snd
        let run fs = List.map (fun f -> f()) fs |> toDV

        let bench =
            [ if bench1  then
                if auto then  yield run row_AD ./ run row_originalsD
                if numeric then yield  run row_N  ./ run row_originals
              if bench2 then
                if auto then yield run row'_AD ./ run row_originalsD
                if numeric then yield run row'_N  ./ run row_originals ]
            |> DM.ofRows

        let score = float (DM.sum bench)

        let score = 1. / score
        let score = int (score * 100000000.)

        printfn "Benchmark score: %A\n" score

        printfn "Writing results to file: %s" fileName

        let stream = new System.IO.StreamWriter(fileName, false)
        stream.WriteLine("DiffSharp Benchmarks")
        stream.WriteLine("Copyright (c) 2014--2016, National University of Ireland Maynooth.")
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

        stream.WriteLine("Benchmark matrix\r\n")
        stream.WriteLine("Column labels: {0}",String.concat ", " opNames2 )
        stream.WriteLine("Row labels: {0}", String.concat ", " [ if bench1 then
                                                                    if auto then yield "DiffSharp.Bench1.AD"
                                                                    if numeric then yield "Diffsharp.Bench1.Numerical"
                                                                 if bench2 then
                                                                    if auto then yield "DiffSharp.Bench2.AD"
                                                                    if numeric then yield "Diffsharp.Bench2.Numerical"
                                                                 ])
        stream.WriteLine(sprintf "Values: %s" (bench.ToMathematicaString()))

        stream.Flush()
        stream.Close()


        0
