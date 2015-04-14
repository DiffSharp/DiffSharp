//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
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
open FsAlg.Generic

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

let printArray (s:System.IO.StreamWriter) (o:obj[]) =
    for a in o do
        match a with
        | :? (float[]) as f -> s.WriteLine((vector f).ToString())
        | :? (float[,]) as f -> s.WriteLine((Matrix.ofArray2D f).ToString())
        | _ -> s.WriteLine(a.ToString())

let printb i t name =
    printfn "Running benchmark %2i/%2i %s ..." i  t name

type options = {
    repetitions : int;
    fileName : string;
    help : bool;
    changed : bool;
    }

let minRepetitions = 1000

let dateTimeString (d:System.DateTime) =
    sprintf "%s%s%s%s%s%s" (d.Year.ToString()) (d.Month.ToString("D2")) (d.Day.ToString("D2")) (d.Hour.ToString("D2")) (d.Minute.ToString("D2")) (d.Second.ToString("D2"))

let defaultOptions = {
    repetitions = 50000; // > 100000 seems to work fine
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
                eprintfn "Option -r was followed by an invalid input."
                parseArgsRec xs optionsSoFar
        | _ ->
            eprintfn "Option -r needs to be followed by a number."
            parseArgsRec xs optionsSoFar
    | x::xs ->
        eprintfn "Option \"%s\" is unrecognized." x
        parseArgsRec xs optionsSoFar

let parseArgs args =

    parseArgsRec args defaultOptions

[<EntryPoint>]
let main argv = 

    let benchmarkver = "1.0.5"

    printfn "DiffSharp Benchmarks"

    printfn "Copyright (c) 2014, 2015, National University of Ireland Maynooth."
    printfn "Written by: Atilim Gunes Baydin, Barak A. Pearlmutter\n"

    let ops = parseArgs (List.ofArray argv)

    if ops.help then
        printfn "Runs a series of benchmarks testing the operations in the DiffSharp library.\n"
        printfn "dsbench [-r repetitions] [-f filename]\n"
        printfn "  -r repetitions  Specifies the number of repetitions."
        printfn "                  Higher values give more accurate results, through averaging."
        printfn "                  Default: %i" defaultOptions.repetitions
        printfn "                  Minimum:  %i" minRepetitions
        printfn "  -f filename     Specifies the name of the output file."
        printfn "                  If the file exists, it will be overwritten."
        printfn "                  Default: DiffSharpBenchmark + current time + .txt"
        0 // return an integer exit code
    else
        printfn "Use option -h for help on usage.\n"
    
        if not ops.changed then printfn "Using default options.\n"

        let n = ops.repetitions
        let nsymbolic = n / 1000
        let noriginal = n * 100
        let fileName = ops.fileName

        printfn "Repetitions: %A" n
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

        let x = 2.8
        let xv = [|2.2; 3.5; 5.1|]
        let v = [|1.2; 3.4; 5.2|]
        let u = [|1.5; 3.1; 5.4|]

        printb 1 35 "original functions"
        let res_fss, dur_fss = duration noriginal (fun () -> (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_fvs, dur_fvs = duration noriginal (fun () -> (fun (x:float[]) -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_fvv, dur_fvv = duration noriginal (fun () -> (fun (x:float[]) -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        if dur_fss = 0. || dur_fvs = 0. || dur_fvv = 0. then printfn "***\n WARNING: Zero duration encountered for an original function\n***"

        printb 2 35 "diff"
        let res_diff_AD_Forward, dur_diff_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_Forward2, dur_diff_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_ForwardG, dur_diff_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_ForwardGH, dur_diff_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_ForwardN, dur_diff_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_ForwardReverse, dur_diff_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_AD_Reverse, dur_diff_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_Numerical, dur_diff_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SymbolicCompile, dur_diff_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff_Symbolic = DiffSharp.Symbolic.DiffOps.diff <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff_SymbolicUse, dur_diff_SymbolicUse = duration nsymbolic (fun () -> f_diff_Symbolic x)

        printb 3 35 "diff2"
        let res_diff2_AD_Forward, dur_diff2_AD_Forward = 0., 0.
        let res_diff2_AD_Forward2, dur_diff2_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_AD_ForwardG, dur_diff2_AD_ForwardG = 0., 0.
        let res_diff2_AD_ForwardGH, dur_diff2_AD_ForwardGH = 0., 0.
        let res_diff2_AD_ForwardN, dur_diff2_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_AD_ForwardReverse, dur_diff2_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_AD_Reverse, dur_diff2_AD_Reverse = 0., 0.
        let res_diff2_Numerical, dur_diff2_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_SymbolicCompile, dur_diff2_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff2_Symbolic = DiffSharp.Symbolic.DiffOps.diff2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff2_SymbolicUse, dur_diff2_SymbolicUse = duration nsymbolic (fun () -> f_diff2_Symbolic x)

        printb 4 35 "diffn"
        let res_diffn_AD_Forward, dur_diffn_AD_Forward = 0., 0.
        let res_diffn_AD_Forward2, dur_diffn_AD_Forward2 = 0., 0.
        let res_diffn_AD_ForwardG, dur_diffn_AD_ForwardG = 0., 0.
        let res_diffn_AD_ForwardGH, dur_diffn_AD_ForwardGH = 0., 0.
        let res_diffn_AD_ForwardN, dur_diffn_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diffn_AD_ForwardReverse, dur_diffn_AD_ForwardReverse = 0., 0.
        let res_diffn_AD_Reverse, dur_diffn_AD_Reverse = 0., 0.
        let res_diffn_Numerical, dur_diffn_Numerical = 0., 0.
        let res_diffn_SymbolicCompile, dur_diffn_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diffn 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diffn_Symbolic = DiffSharp.Symbolic.DiffOps.diffn 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diffn_SymbolicUse, dur_diffn_SymbolicUse = duration nsymbolic (fun () -> f_diffn_Symbolic x)

        printb 5 35 "grad"
        let res_grad_AD_Forward, dur_grad_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_Forward2, dur_grad_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_ForwardG, dur_grad_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_ForwardGH, dur_grad_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_ForwardN, dur_grad_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.grad (fun x ->(x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_ForwardReverse, dur_grad_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.grad (fun x ->(x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_AD_Reverse, dur_grad_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_Numerical, dur_grad_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SymbolicCompile, dur_grad_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.grad <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_grad_Symbolic = DiffSharp.Symbolic.DiffOps.grad <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_grad_SymbolicUse, dur_grad_SymbolicUse = duration nsymbolic (fun () -> f_grad_Symbolic xv)

        printb 6 35 "gradv"
        let res_gradv_AD_Forward, dur_gradv_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_AD_Forward2, dur_gradv_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_AD_ForwardG, dur_gradv_AD_ForwardG = 0., 0.
        let res_gradv_AD_ForwardGH, dur_gradv_AD_ForwardGH = 0., 0.
        let res_gradv_AD_ForwardN, dur_gradv_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_AD_ForwardReverse, dur_gradv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_AD_Reverse, dur_gradv_AD_Reverse = 0., 0.
        let res_gradv_Numerical, dur_gradv_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_SymbolicCompile, dur_gradv_SymbolicCompile = 0., 0.
        let res_gradv_SymbolicUse, dur_gradv_SymbolicUse = 0., 0.

        printb 7 35 "hessian"
        let res_hessian_AD_Forward, dur_hessian_AD_Forward = 0., 0.
        let res_hessian_AD_Forward2, dur_hessian_AD_Forward2 = 0., 0.
        let res_hessian_AD_ForwardG, dur_hessian_AD_ForwardG = 0., 0.
        let res_hessian_AD_ForwardGH, dur_hessian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_AD_ForwardN, dur_hessian_AD_ForwardN = 0., 0.
        let res_hessian_AD_ForwardReverse, dur_hessian_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_AD_Reverse, dur_hessian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_Numerical, dur_hessian_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_SymbolicCompile, dur_hessian_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.hessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_hessian_Symbolic = DiffSharp.Symbolic.DiffOps.hessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_hessian_SymbolicUse, dur_hessian_SymbolicUse = duration nsymbolic (fun () -> f_hessian_Symbolic xv)

        printb 8 35 "hessianv"
        let res_hessianv_AD_Forward, dur_hessianv_AD_Forward = 0., 0.
        let res_hessianv_AD_Forward2, dur_hessianv_AD_Forward2 = 0., 0.
        let res_hessianv_AD_ForwardG, dur_hessianv_AD_ForwardG = 0., 0.
        let res_hessianv_AD_ForwardGH, dur_hessianv_AD_ForwardGH = 0., 0.
        let res_hessianv_AD_ForwardN, dur_hessianv_AD_ForwardN = 0., 0.
        let res_hessianv_AD_ForwardReverse, dur_hessianv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.hessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv_AD_Reverse, dur_hessianv_AD_Reverse = 0., 0.
        let res_hessianv_Numerical, dur_hessianv_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv_SymbolicCompile, dur_hessianv_SymbolicCompile = 0., 0.
        let res_hessianv_SymbolicUse, dur_hessianv_SymbolicUse = 0., 0.

        printb 9 35 "gradhessian"
        let res_gradhessian_AD_Forward, dur_gradhessian_AD_Forward = 0., 0.
        let res_gradhessian_AD_Forward2, dur_gradhessian_AD_Forward2 = 0., 0.
        let res_gradhessian_AD_ForwardG, dur_gradhessian_AD_ForwardG = 0., 0.
        let res_gradhessian_AD_ForwardGH, dur_gradhessian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_AD_ForwardN, dur_gradhessian_AD_ForwardN = 0., 0.
        let res_gradhessian_AD_ForwardReverse, dur_gradhessian_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_AD_Reverse, dur_gradhessian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_Numerical, dur_gradhessian_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_SymbolicCompile, dur_gradhessian_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.gradhessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_gradhessian_Symbolic = DiffSharp.Symbolic.DiffOps.gradhessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_gradhessian_SymbolicUse, dur_gradhessian_SymbolicUse = duration nsymbolic (fun () -> f_gradhessian_Symbolic xv)

        printb 10 35 "gradhessianv"
        let res_gradhessianv_AD_Forward, dur_gradhessianv_AD_Forward = 0., 0.
        let res_gradhessianv_AD_Forward2, dur_gradhessianv_AD_Forward2 = 0., 0.
        let res_gradhessianv_AD_ForwardG, dur_gradhessianv_AD_ForwardG = 0., 0.
        let res_gradhessianv_AD_ForwardGH, dur_gradhessianv_AD_ForwardGH = 0., 0.
        let res_gradhessianv_AD_ForwardN, dur_gradhessianv_AD_ForwardN = 0., 0.
        let res_gradhessianv_AD_ForwardReverse, dur_gradhessianv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradhessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv_AD_Reverse, dur_gradhessianv_AD_Reverse = 0., 0.
        let res_gradhessianv_Numerical, dur_gradhessianv_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv_SymbolicCompile, dur_gradhessianv_SymbolicCompile = 0., 0.
        let res_gradhessianv_SymbolicUse, dur_gradhessianv_SymbolicUse = 0., 0.

        printb 11 35 "laplacian"
        let res_laplacian_AD_Forward, dur_laplacian_AD_Forward = 0., 0.
        let res_laplacian_AD_Forward2, dur_laplacian_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_AD_ForwardG, dur_laplacian_AD_ForwardG = 0., 0.
        let res_laplacian_AD_ForwardGH, dur_laplacian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_AD_ForwardN, dur_laplacian_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_AD_ForwardReverse, dur_laplacian_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_AD_Reverse, dur_laplacian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_Numerical, dur_laplacian_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_SymbolicCompile, dur_laplacian_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.laplacian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_laplacian_Symbolic = DiffSharp.Symbolic.DiffOps.laplacian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_laplacian_SymbolicUse, dur_laplacian_SymbolicUse = duration nsymbolic (fun () -> f_laplacian_Symbolic xv)

        printb 12 35 "jacobian"
        let res_jacobian_AD_Forward, dur_jacobian_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_Forward2, dur_jacobian_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_ForwardG, dur_jacobian_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_ForwardGH, dur_jacobian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_ForwardN, dur_jacobian_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_ForwardReverse, dur_jacobian_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_AD_Reverse, dur_jacobian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_Numerical, dur_jacobian_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SymbolicCompile, dur_jacobian_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobian <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobian_Symbolic = DiffSharp.Symbolic.DiffOps.jacobian <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobian_SymbolicUse, dur_jacobian_SymbolicUse = duration nsymbolic (fun () -> f_jacobian_Symbolic xv)

        printb 13 35 "jacobianv"
        let res_jacobianv_AD_Forward, dur_jacobianv_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_AD_Forward2, dur_jacobianv_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_AD_ForwardG, dur_jacobianv_AD_ForwardG =  0., 0.
        let res_jacobianv_AD_ForwardGH, dur_jacobianv_AD_ForwardGH = 0., 0.
        let res_jacobianv_AD_ForwardN, dur_jacobianv_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_AD_ForwardReverse, dur_jacobianv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_AD_Reverse, dur_jacobianv_AD_Reverse = 0., 0.
        let res_jacobianv_Numerical, dur_jacobianv_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_SymbolicCompile, dur_jacobianv_SymbolicCompile =  0., 0.
        let res_jacobianv_SymbolicUse, dur_jacobianv_SymbolicUse =  0., 0.

        printb 14 35 "jacobianT"
        let res_jacobianT_AD_Forward, dur_jacobianT_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_Forward2, dur_jacobianT_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_ForwardG, dur_jacobianT_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_ForwardGH, dur_jacobianT_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_ForwardN, dur_jacobianT_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_ForwardReverse, dur_jacobianT_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_AD_Reverse, dur_jacobianT_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_Numerical, dur_jacobianT_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SymbolicCompile, dur_jacobianT_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobianT <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobianT_Symbolic = DiffSharp.Symbolic.DiffOps.jacobianT <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobianT_SymbolicUse, dur_jacobianT_SymbolicUse = duration nsymbolic (fun () -> f_jacobianT_Symbolic xv)

        printb 15 35 "jacobianTv"
        let res_jacobianTv_AD_Forward, dur_jacobianTv_AD_Forward = 0., 0.
        let res_jacobianTv_AD_Forward2, dur_jacobianTv_AD_Forward2 = 0., 0.
        let res_jacobianTv_AD_ForwardG, dur_jacobianTv_AD_ForwardG =  0., 0.
        let res_jacobianTv_AD_ForwardGH, dur_jacobianTv_AD_ForwardGH = 0., 0.
        let res_jacobianTv_AD_ForwardN, dur_jacobianTv_AD_ForwardN = 0., 0.
        let res_jacobianTv_AD_ForwardReverse, dur_jacobianTv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv_AD_Reverse, dur_jacobianTv_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv_Numerical, dur_jacobianTv_Numerical = 0., 0.
        let res_jacobianTv_SymbolicCompile, dur_jacobianTv_SymbolicCompile = 0., 0.
        let res_jacobianTv_SymbolicUse, dur_jacobianTv_SymbolicUse = 0., 0.

        printb 16 35 "curl"
        let res_curl_AD_Forward, dur_curl_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_Forward2, dur_curl_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_ForwardG, dur_curl_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_ForwardGH, dur_curl_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_ForwardN, dur_curl_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_ForwardReverse, dur_curl_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_AD_Reverse, dur_curl_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_Numerical, dur_curl_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SymbolicCompile, dur_curl_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curl <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curl_Symbolic = DiffSharp.Symbolic.DiffOps.curl <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curl_SymbolicUse, dur_curl_SymbolicUse = duration nsymbolic (fun () -> f_curl_Symbolic xv)

        printb 17 35 "div"
        let res_div_AD_Forward, dur_div_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_Forward2, dur_div_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_ForwardG, dur_div_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_ForwardGH, dur_div_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_ForwardN, dur_div_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_ForwardReverse, dur_div_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_AD_Reverse, dur_div_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_Numerical, dur_div_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SymbolicCompile, dur_div_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.div <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_div_Symbolic = DiffSharp.Symbolic.DiffOps.div <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_div_SymbolicUse, dur_div_SymbolicUse = duration nsymbolic (fun () -> f_div_Symbolic xv)
        
        printb 18 35 "curldiv"
        let res_curldiv_AD_Forward, dur_curldiv_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_Forward2, dur_curldiv_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_ForwardG, dur_curldiv_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_ForwardGH, dur_curldiv_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_ForwardN, dur_curldiv_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_ForwardReverse, dur_curldiv_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_AD_Reverse, dur_curldiv_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_Numerical, dur_curldiv_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SymbolicCompile, dur_curldiv_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curldiv <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curldiv_Symbolic = DiffSharp.Symbolic.DiffOps.curldiv <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curldiv_SymbolicUse, dur_curldiv_SymbolicUse = duration nsymbolic (fun () -> f_curldiv_Symbolic xv)

        //
        //
        //
        //
        //

        printb 19 35 "diff'"
        let res_diff'_AD_Forward, dur_diff'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_Forward2, dur_diff'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_ForwardG, dur_diff'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_ForwardGH, dur_diff'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_ForwardN, dur_diff'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_ForwardReverse, dur_diff'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_AD_Reverse, dur_diff'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_Numerical, dur_diff'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SymbolicCompile, dur_diff'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff'_Symbolic = DiffSharp.Symbolic.DiffOps.diff' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff'_SymbolicUse, dur_diff'_SymbolicUse = duration nsymbolic (fun () -> f_diff'_Symbolic x)

        printb 20 35 "diff2'"
        let res_diff2'_AD_Forward, dur_diff2'_AD_Forward = 0., 0.
        let res_diff2'_AD_Forward2, dur_diff2'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_AD_ForwardG, dur_diff2'_AD_ForwardG = 0., 0.
        let res_diff2'_AD_ForwardGH, dur_diff2'_AD_ForwardGH = 0., 0.
        let res_diff2'_AD_ForwardN, dur_diff2'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_AD_ForwardReverse, dur_diff2'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_AD_Reverse, dur_diff2'_AD_Reverse = 0., 0.
        let res_diff2'_Numerical, dur_diff2'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_SymbolicCompile, dur_diff2'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff2' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff2'_Symbolic = DiffSharp.Symbolic.DiffOps.diff2' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff2'_SymbolicUse, dur_diff2'_SymbolicUse = duration nsymbolic (fun () -> f_diff2'_Symbolic x)

        printb 21 35 "diffn'"
        let res_diffn'_AD_Forward, dur_diffn'_AD_Forward = 0., 0.
        let res_diffn'_AD_Forward2, dur_diffn'_AD_Forward2 = 0., 0.
        let res_diffn'_AD_ForwardG, dur_diffn'_AD_ForwardG = 0., 0.
        let res_diffn'_AD_ForwardGH, dur_diffn'_AD_ForwardGH = 0., 0.
        let res_diffn'_AD_ForwardN, dur_diffn'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diffn'_AD_ForwardReverse, dur_diffn'_AD_ForwardReverse = 0., 0.
        let res_diffn'_AD_Reverse, dur_diffn'_AD_Reverse = 0., 0.
        let res_diffn'_Numerical, dur_diffn'_Numerical = 0., 0.
        let res_diffn'_SymbolicCompile, dur_diffn'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diffn' 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diffn'_Symbolic = DiffSharp.Symbolic.DiffOps.diffn' 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diffn'_SymbolicUse, dur_diffn'_SymbolicUse = duration nsymbolic (fun () -> f_diffn'_Symbolic x)

        printb 22 35 "grad'"
        let res_grad'_AD_Forward, dur_grad'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_Forward2, dur_grad'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_ForwardG, dur_grad'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_ForwardGH, dur_grad'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_ForwardN, dur_grad'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_ForwardReverse, dur_grad'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_AD_Reverse, dur_grad'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_Numerical, dur_grad'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SymbolicCompile, dur_grad'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.grad' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_grad'_Symbolic = DiffSharp.Symbolic.DiffOps.grad' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_grad'_SymbolicUse, dur_grad'_SymbolicUse = duration nsymbolic (fun () -> f_grad'_Symbolic xv)

        printb 23 35 "gradv'"
        let res_gradv'_AD_Forward, dur_gradv'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_AD_Forward2, dur_gradv'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_AD_ForwardG, dur_gradv'_AD_ForwardG = 0., 0.
        let res_gradv'_AD_ForwardGH, dur_gradv'_AD_ForwardGH = 0., 0.
        let res_gradv'_AD_ForwardN, dur_gradv'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_AD_ForwardReverse, dur_gradv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_AD_Reverse, dur_gradv'_AD_Reverse = 0., 0.
        let res_gradv'_Numerical, dur_gradv'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_SymbolicCompile, dur_gradv'_SymbolicCompile = 0., 0.
        let res_gradv'_SymbolicUse, dur_gradv'_SymbolicUse = 0., 0.

        printb 24 35 "hessian'"
        let res_hessian'_AD_Forward, dur_hessian'_AD_Forward = 0., 0.
        let res_hessian'_AD_Forward2, dur_hessian'_AD_Forward2 = 0., 0.
        let res_hessian'_AD_ForwardG, dur_hessian'_AD_ForwardG = 0., 0.
        let res_hessian'_AD_ForwardGH, dur_hessian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_AD_ForwardN, dur_hessian'_AD_ForwardN = 0., 0.
        let res_hessian'_AD_ForwardReverse, dur_hessian'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_AD_Reverse, dur_hessian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_Numerical, dur_hessian'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_SymbolicCompile, dur_hessian'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.hessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_hessian'_Symbolic = DiffSharp.Symbolic.DiffOps.hessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_hessian'_SymbolicUse, dur_hessian'_SymbolicUse = duration nsymbolic (fun () -> f_hessian'_Symbolic xv)

        printb 25 35 "hessianv'"
        let res_hessianv'_AD_Forward, dur_hessianv'_AD_Forward = 0., 0.
        let res_hessianv'_AD_Forward2, dur_hessianv'_AD_Forward2 = 0., 0.
        let res_hessianv'_AD_ForwardG, dur_hessianv'_AD_ForwardG = 0., 0.
        let res_hessianv'_AD_ForwardGH, dur_hessianv'_AD_ForwardGH = 0., 0.
        let res_hessianv'_AD_ForwardN, dur_hessianv'_AD_ForwardN = 0., 0.
        let res_hessianv'_AD_ForwardReverse, dur_hessianv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.hessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv'_AD_Reverse, dur_hessianv'_AD_Reverse = 0., 0.
        let res_hessianv'_Numerical, dur_hessianv'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv'_SymbolicCompile, dur_hessianv'_SymbolicCompile = 0., 0.
        let res_hessianv'_SymbolicUse, dur_hessianv'_SymbolicUse = 0., 0.

        printb 26 35 "gradhessian'"
        let res_gradhessian'_AD_Forward, dur_gradhessian'_AD_Forward = 0., 0.
        let res_gradhessian'_AD_Forward2, dur_gradhessian'_AD_Forward2 = 0., 0.
        let res_gradhessian'_AD_ForwardG, dur_gradhessian'_AD_ForwardG = 0., 0.
        let res_gradhessian'_AD_ForwardGH, dur_gradhessian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_AD_ForwardN, dur_gradhessian'_AD_ForwardN = 0., 0.
        let res_gradhessian'_AD_ForwardReverse, dur_gradhessian'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_AD_Reverse, dur_gradhessian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_Numerical, dur_gradhessian'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_SymbolicCompile, dur_gradhessian'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.gradhessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_gradhessian'_Symbolic = DiffSharp.Symbolic.DiffOps.gradhessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_gradhessian'_SymbolicUse, dur_gradhessian'_SymbolicUse = duration nsymbolic (fun () -> f_gradhessian'_Symbolic xv)

        printb 27 35 "gradhessianv'"
        let res_gradhessianv'_AD_Forward, dur_gradhessianv'_AD_Forward = 0., 0.
        let res_gradhessianv'_AD_Forward2, dur_gradhessianv'_AD_Forward2 = 0., 0.
        let res_gradhessianv'_AD_ForwardG, dur_gradhessianv'_AD_ForwardG = 0., 0.
        let res_gradhessianv'_AD_ForwardGH, dur_gradhessianv'_AD_ForwardGH = 0., 0.
        let res_gradhessianv'_AD_ForwardN, dur_gradhessianv'_AD_ForwardN = 0., 0.
        let res_gradhessianv'_AD_ForwardReverse, dur_gradhessianv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.gradhessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv'_AD_Reverse, dur_gradhessianv'_AD_Reverse = 0., 0.
        let res_gradhessianv'_Numerical, dur_gradhessianv'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv'_SymbolicCompile, dur_gradhessianv'_SymbolicCompile = 0., 0.
        let res_gradhessianv'_SymbolicUse, dur_gradhessianv'_SymbolicUse = 0., 0.

        printb 28 35 "laplacian'"
        let res_laplacian'_AD_Forward, dur_laplacian'_AD_Forward = 0., 0.
        let res_laplacian'_AD_Forward2, dur_laplacian'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_AD_ForwardG, dur_laplacian'_AD_ForwardG = 0., 0.
        let res_laplacian'_AD_ForwardGH, dur_laplacian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_AD_ForwardN, dur_laplacian'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_AD_ForwardReverse, dur_laplacian'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_AD_Reverse, dur_laplacian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_Numerical, dur_laplacian'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_SymbolicCompile, dur_laplacian'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.laplacian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_laplacian'_Symbolic = DiffSharp.Symbolic.DiffOps.laplacian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_laplacian'_SymbolicUse, dur_laplacian'_SymbolicUse = duration nsymbolic (fun () -> f_laplacian'_Symbolic xv)

        printb 29 35 "jacobian'"
        let res_jacobian'_AD_Forward, dur_jacobian'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_Forward2, dur_jacobian'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_ForwardG, dur_jacobian'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_ForwardGH, dur_jacobian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_ForwardN, dur_jacobian'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_ForwardReverse, dur_jacobian'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_AD_Reverse, dur_jacobian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_Numerical, dur_jacobian'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SymbolicCompile, dur_jacobian'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobian' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobian'_Symbolic = DiffSharp.Symbolic.DiffOps.jacobian' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobian'_SymbolicUse, dur_jacobian'_SymbolicUse = duration nsymbolic (fun () -> f_jacobian'_Symbolic xv)

        printb 30 35 "jacobianv'"
        let res_jacobianv'_AD_Forward, dur_jacobianv'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_AD_Forward2, dur_jacobianv'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_AD_ForwardG, dur_jacobianv'_AD_ForwardG = 0., 0.
        let res_jacobianv'_AD_ForwardGH, dur_jacobianv'_AD_ForwardGH = 0., 0.
        let res_jacobianv'_AD_ForwardN, dur_jacobianv'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_AD_ForwardReverse, dur_jacobianv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_AD_Reverse, dur_jacobianv'_AD_Reverse = 0., 0.
        let res_jacobianv'_Numerical, dur_jacobianv'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_SymbolicCompile, dur_jacobianv'_SymbolicCompile = 0., 0.
        let res_jacobianv'_SymbolicUse, dur_jacobianv'_SymbolicUse = 0., 0.

        printb 31 35 "jacobianT'"
        let res_jacobianT'_AD_Forward, dur_jacobianT'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_Forward2, dur_jacobianT'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_ForwardG, dur_jacobianT'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_ForwardGH, dur_jacobianT'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_ForwardN, dur_jacobianT'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_ForwardReverse, dur_jacobianT'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_AD_Reverse, dur_jacobianT'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_Numerical, dur_jacobianT'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SymbolicCompile, dur_jacobianT'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobianT' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobianT'_Symbolic = DiffSharp.Symbolic.DiffOps.jacobianT' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobianT'_SymbolicUse, dur_jacobianT'_SymbolicUse = duration nsymbolic (fun () -> f_jacobianT'_Symbolic xv)

        printb 32 35 "jacobianTv'"
        let res_jacobianTv'_AD_Forward, dur_jacobianTv'_AD_Forward = 0., 0.
        let res_jacobianTv'_AD_Forward2, dur_jacobianTv'_AD_Forward2 = 0., 0.
        let res_jacobianTv'_AD_ForwardG, dur_jacobianTv'_AD_ForwardG = 0., 0.
        let res_jacobianTv'_AD_ForwardGH, dur_jacobianTv'_AD_ForwardGH = 0., 0.
        let res_jacobianTv'_AD_ForwardN, dur_jacobianTv'_AD_ForwardN = 0., 0.
        let res_jacobianTv'_AD_ForwardReverse, dur_jacobianTv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv'_AD_Reverse, dur_jacobianTv'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv'_Numerical, dur_jacobianTv'_Numerical = 0., 0.
        let res_jacobianTv'_SymbolicCompile, dur_jacobianTv'_SymbolicCompile = 0., 0.
        let res_jacobianTv'_SymbolicUse, dur_jacobianTv'_SymbolicUse = 0., 0.

        printb 33 35 "curl'"
        let res_curl'_AD_Forward, dur_curl'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_Forward2, dur_curl'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_ForwardG, dur_curl'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_ForwardGH, dur_curl'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_ForwardN, dur_curl'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_ForwardReverse, dur_curl'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_AD_Reverse, dur_curl'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_Numerical, dur_curl'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SymbolicCompile, dur_curl'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curl' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curl'_Symbolic = DiffSharp.Symbolic.DiffOps.curl' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curl'_SymbolicUse, dur_curl'_SymbolicUse = duration nsymbolic (fun () -> f_curl'_Symbolic xv)

        printb 34 35 "div'"
        let res_div'_AD_Forward, dur_div'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_Forward2, dur_div'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_ForwardG, dur_div'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_ForwardGH, dur_div'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_ForwardN, dur_div'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_ForwardReverse, dur_div'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_AD_Reverse, dur_div'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_Numerical, dur_div'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SymbolicCompile, dur_div'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.div' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_div'_Symbolic = DiffSharp.Symbolic.DiffOps.div' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_div'_SymbolicUse, dur_div'_SymbolicUse = duration nsymbolic (fun () -> f_div'_Symbolic xv)

        printb 35 35 "curldiv'"
        let res_curldiv'_AD_Forward, dur_curldiv'_AD_Forward = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_Forward2, dur_curldiv'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_ForwardG, dur_curldiv'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_ForwardGH, dur_curldiv'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_ForwardN, dur_curldiv'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_ForwardReverse, dur_curldiv'_AD_ForwardReverse = duration n (fun () -> DiffSharp.AD.Specialized.Forward1Reverse1.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_AD_Reverse, dur_curldiv'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_Numerical, dur_curldiv'_Numerical = duration n (fun () -> DiffSharp.Numerical.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SymbolicCompile, dur_curldiv'_SymbolicCompile = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curldiv' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curldiv'_Symbolic = DiffSharp.Symbolic.DiffOps.curldiv' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curldiv'_SymbolicUse, dur_curldiv'_SymbolicUse = duration nsymbolic (fun () -> f_curldiv'_Symbolic xv)

        //
        //
        //
        //
        //
        //

        let finished = System.DateTime.Now
        let duration = finished - started
        printfn "Benchmarking finished: %A\n" finished
        printfn "Total duration: %A\n" duration

        let row_originals = vector [dur_fss; dur_fss; dur_fss; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv]
        let row_AD_Forward = vector [dur_diff_AD_Forward; dur_diff2_AD_Forward; dur_diffn_AD_Forward; dur_grad_AD_Forward; dur_gradv_AD_Forward; dur_hessian_AD_Forward; dur_hessianv_AD_Forward; dur_gradhessian_AD_Forward; dur_gradhessianv_AD_Forward; dur_laplacian_AD_Forward; dur_jacobian_AD_Forward; dur_jacobianv_AD_Forward; dur_jacobianT_AD_Forward; dur_jacobianTv_AD_Forward; dur_curl_AD_Forward; dur_div_AD_Forward; dur_curldiv_AD_Forward]
        let row_AD_Forward2 = vector [dur_diff_AD_Forward2; dur_diff2_AD_Forward2; dur_diffn_AD_Forward2; dur_grad_AD_Forward2; dur_gradv_AD_Forward2; dur_hessian_AD_Forward2; dur_hessianv_AD_Forward2; dur_gradhessian_AD_Forward2; dur_gradhessianv_AD_Forward2; dur_laplacian_AD_Forward2; dur_jacobian_AD_Forward2; dur_jacobianv_AD_Forward2; dur_jacobianT_AD_Forward2; dur_jacobianTv_AD_Forward2; dur_curl_AD_Forward2; dur_div_AD_Forward2; dur_curldiv_AD_Forward2]
        let row_AD_ForwardG = vector [dur_diff_AD_ForwardG; dur_diff2_AD_ForwardG; dur_diffn_AD_ForwardG; dur_grad_AD_ForwardG; dur_gradv_AD_ForwardG; dur_hessian_AD_ForwardG; dur_hessianv_AD_ForwardG; dur_gradhessian_AD_ForwardG; dur_gradhessianv_AD_ForwardG; dur_laplacian_AD_ForwardG; dur_jacobian_AD_ForwardG; dur_jacobianv_AD_ForwardG; dur_jacobianT_AD_ForwardG; dur_jacobianTv_AD_ForwardG; dur_curl_AD_ForwardG; dur_div_AD_ForwardG; dur_curldiv_AD_ForwardG]
        let row_AD_ForwardGH = vector [dur_diff_AD_ForwardGH; dur_diff2_AD_ForwardGH; dur_diffn_AD_ForwardGH; dur_grad_AD_ForwardGH; dur_gradv_AD_ForwardGH; dur_hessian_AD_ForwardGH; dur_hessianv_AD_ForwardGH; dur_gradhessian_AD_ForwardGH; dur_gradhessianv_AD_ForwardGH; dur_laplacian_AD_ForwardGH; dur_jacobian_AD_ForwardGH; dur_jacobianv_AD_ForwardGH; dur_jacobianT_AD_ForwardGH; dur_jacobianTv_AD_ForwardGH; dur_curl_AD_ForwardGH; dur_div_AD_ForwardGH; dur_curldiv_AD_ForwardGH]
        let row_AD_ForwardN = vector [dur_diff_AD_ForwardN; dur_diff2_AD_ForwardN; dur_diffn_AD_ForwardN; dur_grad_AD_ForwardN; dur_gradv_AD_ForwardN; dur_hessian_AD_ForwardN; dur_hessianv_AD_ForwardN; dur_gradhessian_AD_ForwardN; dur_gradhessianv_AD_ForwardN; dur_laplacian_AD_ForwardN; dur_jacobian_AD_ForwardN; dur_jacobianv_AD_ForwardN; dur_jacobianT_AD_ForwardN; dur_jacobianTv_AD_ForwardN; dur_curl_AD_ForwardN; dur_div_AD_ForwardN; dur_curldiv_AD_ForwardN]
        let row_AD_ForwardReverse = vector [dur_diff_AD_ForwardReverse; dur_diff2_AD_ForwardReverse; dur_diffn_AD_ForwardReverse; dur_grad_AD_ForwardReverse; dur_gradv_AD_ForwardReverse; dur_hessian_AD_ForwardReverse; dur_hessianv_AD_ForwardReverse; dur_gradhessian_AD_ForwardReverse; dur_gradhessianv_AD_ForwardReverse; dur_laplacian_AD_ForwardReverse; dur_jacobian_AD_ForwardReverse; dur_jacobianv_AD_ForwardReverse; dur_jacobianT_AD_ForwardReverse; dur_jacobianTv_AD_ForwardReverse; dur_curl_AD_ForwardReverse; dur_div_AD_ForwardReverse; dur_curldiv_AD_ForwardReverse]
        let row_AD_Reverse = vector [dur_diff_AD_Reverse; dur_diff2_AD_Reverse; dur_diffn_AD_Reverse; dur_grad_AD_Reverse; dur_gradv_AD_Reverse; dur_hessian_AD_Reverse; dur_hessianv_AD_Reverse; dur_gradhessian_AD_Reverse; dur_gradhessianv_AD_Reverse; dur_laplacian_AD_Reverse; dur_jacobian_AD_Reverse; dur_jacobianv_AD_Reverse; dur_jacobianT_AD_Reverse; dur_jacobianTv_AD_Reverse; dur_curl_AD_Reverse; dur_div_AD_Reverse; dur_curldiv_AD_Reverse]
        let row_Numerical = vector [dur_diff_Numerical; dur_diff2_Numerical; dur_diffn_Numerical; dur_grad_Numerical; dur_gradv_Numerical; dur_hessian_Numerical; dur_hessianv_Numerical; dur_gradhessian_Numerical; dur_gradhessianv_Numerical; dur_laplacian_Numerical; dur_jacobian_Numerical; dur_jacobianv_Numerical; dur_jacobianT_Numerical; dur_jacobianTv_Numerical; dur_curl_Numerical; dur_div_Numerical; dur_curldiv_Numerical]
        let row_SymbolicCompile = vector [dur_diff_SymbolicCompile; dur_diff2_SymbolicCompile; dur_diffn_SymbolicCompile; dur_grad_SymbolicCompile; dur_gradv_SymbolicCompile; dur_hessian_SymbolicCompile; dur_hessianv_SymbolicCompile; dur_gradhessian_SymbolicCompile; dur_gradhessianv_SymbolicCompile; dur_laplacian_SymbolicCompile; dur_jacobian_SymbolicCompile; dur_jacobianv_SymbolicCompile; dur_jacobianT_SymbolicCompile; dur_jacobianTv_SymbolicCompile; dur_curl_SymbolicCompile; dur_div_SymbolicCompile; dur_curldiv_SymbolicCompile]
        let row_SymbolicUse = vector [dur_diff_SymbolicUse; dur_diff2_SymbolicUse; dur_diffn_SymbolicUse; dur_grad_SymbolicUse; dur_gradv_SymbolicUse; dur_hessian_SymbolicUse; dur_hessianv_SymbolicUse; dur_gradhessian_SymbolicUse; dur_gradhessianv_SymbolicUse; dur_laplacian_SymbolicUse; dur_jacobian_SymbolicUse; dur_jacobianv_SymbolicUse; dur_jacobianT_SymbolicUse; dur_jacobianTv_SymbolicUse; dur_curl_SymbolicUse; dur_div_SymbolicUse; dur_curldiv_SymbolicUse]

        let benchmark = matrix [Vector.toSeq (row_AD_Forward ./ row_originals)
                                Vector.toSeq (row_AD_Forward2 ./ row_originals)
                                Vector.toSeq (row_AD_ForwardG ./ row_originals)
                                Vector.toSeq (row_AD_ForwardGH ./ row_originals)
                                Vector.toSeq (row_AD_ForwardN ./ row_originals)
                                Vector.toSeq (row_AD_ForwardReverse ./ row_originals)
                                Vector.toSeq (row_AD_Reverse ./ row_originals)
                                Vector.toSeq (row_Numerical ./ row_originals)
                                Vector.toSeq (row_SymbolicCompile ./ row_originals)
                                Vector.toSeq (row_SymbolicUse ./ row_originals)]

        let row_AD_Forward' = vector [dur_diff'_AD_Forward; dur_diff2'_AD_Forward; dur_diffn'_AD_Forward; dur_grad'_AD_Forward; dur_gradv'_AD_Forward; dur_hessian'_AD_Forward; dur_hessianv'_AD_Forward; dur_gradhessian'_AD_Forward; dur_gradhessianv'_AD_Forward; dur_laplacian'_AD_Forward; dur_jacobian'_AD_Forward; dur_jacobianv'_AD_Forward; dur_jacobianT'_AD_Forward; dur_jacobianTv'_AD_Forward; dur_curl'_AD_Forward; dur_div'_AD_Forward; dur_curldiv'_AD_Forward]
        let row_AD_Forward2' = vector [dur_diff'_AD_Forward2; dur_diff2'_AD_Forward2; dur_diffn'_AD_Forward2; dur_grad'_AD_Forward2; dur_gradv'_AD_Forward2; dur_hessian'_AD_Forward2; dur_hessianv'_AD_Forward2; dur_gradhessian'_AD_Forward2; dur_gradhessianv'_AD_Forward2; dur_laplacian'_AD_Forward2; dur_jacobian'_AD_Forward2; dur_jacobianv'_AD_Forward2; dur_jacobianT'_AD_Forward2; dur_jacobianTv'_AD_Forward2; dur_curl'_AD_Forward2; dur_div'_AD_Forward2; dur_curldiv'_AD_Forward2]
        let row_AD_ForwardG' = vector [dur_diff'_AD_ForwardG; dur_diff2'_AD_ForwardG; dur_diffn'_AD_ForwardG; dur_grad'_AD_ForwardG; dur_gradv'_AD_ForwardG; dur_hessian'_AD_ForwardG; dur_hessianv'_AD_ForwardG; dur_gradhessian'_AD_ForwardG; dur_gradhessianv'_AD_ForwardG; dur_laplacian'_AD_ForwardG; dur_jacobian'_AD_ForwardG; dur_jacobianv'_AD_ForwardG; dur_jacobianT'_AD_ForwardG; dur_jacobianTv'_AD_ForwardG; dur_curl'_AD_ForwardG; dur_div'_AD_ForwardG; dur_curldiv'_AD_ForwardG]
        let row_AD_ForwardGH' = vector [dur_diff'_AD_ForwardGH; dur_diff2'_AD_ForwardGH; dur_diffn'_AD_ForwardGH; dur_grad'_AD_ForwardGH; dur_gradv'_AD_ForwardGH; dur_hessian'_AD_ForwardGH; dur_hessianv'_AD_ForwardGH; dur_gradhessian'_AD_ForwardGH; dur_gradhessianv'_AD_ForwardGH; dur_laplacian'_AD_ForwardGH; dur_jacobian'_AD_ForwardGH; dur_jacobianv'_AD_ForwardGH; dur_jacobianT'_AD_ForwardGH; dur_jacobianTv'_AD_ForwardGH; dur_curl'_AD_ForwardGH; dur_div'_AD_ForwardGH; dur_curldiv'_AD_ForwardGH]
        let row_AD_ForwardN' = vector [dur_diff'_AD_ForwardN; dur_diff2'_AD_ForwardN; dur_diffn'_AD_ForwardN; dur_grad'_AD_ForwardN; dur_gradv'_AD_ForwardN; dur_hessian'_AD_ForwardN; dur_hessianv'_AD_ForwardN; dur_gradhessian'_AD_ForwardN; dur_gradhessianv'_AD_ForwardN; dur_laplacian'_AD_ForwardN; dur_jacobian'_AD_ForwardN; dur_jacobianv'_AD_ForwardN; dur_jacobianT'_AD_ForwardN; dur_jacobianTv'_AD_ForwardN; dur_curl'_AD_ForwardN; dur_div'_AD_ForwardN; dur_curldiv'_AD_ForwardN]
        let row_AD_ForwardReverse' = vector [dur_diff'_AD_ForwardReverse; dur_diff2'_AD_ForwardReverse; dur_diffn'_AD_ForwardReverse; dur_grad'_AD_ForwardReverse; dur_gradv'_AD_ForwardReverse; dur_hessian'_AD_ForwardReverse; dur_hessianv'_AD_ForwardReverse; dur_gradhessian'_AD_ForwardReverse; dur_gradhessianv'_AD_ForwardReverse; dur_laplacian'_AD_ForwardReverse; dur_jacobian'_AD_ForwardReverse; dur_jacobianv'_AD_ForwardReverse; dur_jacobianT'_AD_ForwardReverse; dur_jacobianTv'_AD_ForwardReverse; dur_curl'_AD_ForwardReverse; dur_div'_AD_ForwardReverse; dur_curldiv'_AD_ForwardReverse]
        let row_AD_Reverse' = vector [dur_diff'_AD_Reverse; dur_diff2'_AD_Reverse; dur_diffn'_AD_Reverse; dur_grad'_AD_Reverse; dur_gradv'_AD_Reverse; dur_hessian'_AD_Reverse; dur_hessianv'_AD_Reverse; dur_gradhessian'_AD_Reverse; dur_gradhessianv'_AD_Reverse; dur_laplacian'_AD_Reverse; dur_jacobian'_AD_Reverse; dur_jacobianv'_AD_Reverse; dur_jacobianT'_AD_Reverse; dur_jacobianTv'_AD_Reverse; dur_curl'_AD_Reverse; dur_div'_AD_Reverse; dur_curldiv'_AD_Reverse]
        let row_Numerical' = vector [dur_diff'_Numerical; dur_diff2'_Numerical; dur_diffn'_Numerical; dur_grad'_Numerical; dur_gradv'_Numerical; dur_hessian'_Numerical; dur_hessianv'_Numerical; dur_gradhessian'_Numerical; dur_gradhessianv'_Numerical; dur_laplacian'_Numerical; dur_jacobian'_Numerical; dur_jacobianv'_Numerical; dur_jacobianT'_Numerical; dur_jacobianTv'_Numerical; dur_curl'_Numerical; dur_div'_Numerical; dur_curldiv'_Numerical]
        let row_SymbolicCompile' = vector [dur_diff'_SymbolicCompile; dur_diff2'_SymbolicCompile; dur_diffn'_SymbolicCompile; dur_grad'_SymbolicCompile; dur_gradv'_SymbolicCompile; dur_hessian'_SymbolicCompile; dur_hessianv'_SymbolicCompile; dur_gradhessian'_SymbolicCompile; dur_gradhessianv'_SymbolicCompile; dur_laplacian'_SymbolicCompile; dur_jacobian'_SymbolicCompile; dur_jacobianv'_SymbolicCompile; dur_jacobianT'_SymbolicCompile; dur_jacobianTv'_SymbolicCompile; dur_curl'_SymbolicCompile; dur_div'_SymbolicCompile; dur_curldiv'_SymbolicCompile]
        let row_SymbolicUse' = vector [dur_diff'_SymbolicUse; dur_diff2'_SymbolicUse; dur_diffn'_SymbolicUse; dur_grad'_SymbolicUse; dur_gradv'_SymbolicUse; dur_hessian'_SymbolicUse; dur_hessianv'_SymbolicUse; dur_gradhessian'_SymbolicUse; dur_gradhessianv'_SymbolicUse; dur_laplacian'_SymbolicUse; dur_jacobian'_SymbolicUse; dur_jacobianv'_SymbolicUse; dur_jacobianT'_SymbolicUse; dur_jacobianTv'_SymbolicUse; dur_curl'_SymbolicUse; dur_div'_SymbolicUse; dur_curldiv'_SymbolicUse]

        let benchmark' = matrix [Vector.toSeq (row_AD_Forward' ./ row_originals)
                                 Vector.toSeq (row_AD_Forward2' ./ row_originals)
                                 Vector.toSeq (row_AD_ForwardG' ./ row_originals)
                                 Vector.toSeq (row_AD_ForwardGH' ./ row_originals)
                                 Vector.toSeq (row_AD_ForwardN' ./ row_originals)
                                 Vector.toSeq (row_AD_ForwardReverse' ./ row_originals)
                                 Vector.toSeq (row_AD_Reverse' ./ row_originals)
                                 Vector.toSeq (row_Numerical' ./ row_originals)
                                 Vector.toSeq (row_SymbolicCompile' ./ row_originals)
                                 Vector.toSeq (row_SymbolicUse' ./ row_originals)]

        let score = (Vector.sum row_AD_Forward) 
                    + (Vector.sum row_AD_Forward2)
                    + (Vector.sum row_AD_ForwardG)
                    + (Vector.sum row_AD_ForwardGH)
                    + (Vector.sum row_AD_ForwardN)
                    + (Vector.sum row_AD_ForwardReverse)
                    + (Vector.sum row_AD_Reverse)
                    + (Vector.sum row_Numerical)
                    + (Vector.sum row_SymbolicCompile)
                    + (Vector.sum row_SymbolicUse)
    
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
    
        stream.WriteLine("Benchmark matrix A\r\n")
        stream.WriteLine("Column labels: {diff, diff2, diffn, grad, gradv, hessian, hessianv, gradhessian, gradhessianv, laplacian, jacobian, jacobianv, jacobianT, jacobianTv, jacobianvTv, curl, div, curldiv}")
        stream.WriteLine("Row labels: {DiffSharp.AD.Forward, DiffSharp.AD.Forward2, DiffSharp.AD.ForwardG, DiffSharp.AD.ForwardGH, DiffSharp.AD.ForwardN, DiffSharp.AD.ForwardReverse, DiffSharp.AD.Reverse, DiffSharp.Numerical, DiffSharp.Symbolic (Compile), DiffSharp.Symbolic (Use)}")
        stream.WriteLine(sprintf "Values: %s" (benchmark.ToMathematicaString()))

        stream.WriteLine("\r\nBenchmark matrix B\r\n")
        stream.WriteLine("Column labels: {diff', diff2', diffn', grad', gradv', hessian', hessianv', gradhessian', gradhessianv', laplacian', jacobian', jacobianv', jacobianT', jacobianTv', jacobianvTv', curl', div', curldiv'}")
        stream.WriteLine("Row labels: {DiffSharp.AD.Forward, DiffSharp.AD.Forward2, DiffSharp.AD.ForwardG, DiffSharp.AD.ForwardGH, DiffSharp.AD.ForwardN, DiffSharp.AD.ForwardReverse, DiffSharp.AD.Reverse, DiffSharp.Numerical, DiffSharp.Symbolic (Compile), DiffSharp.Symbolic (Use)}")
        stream.WriteLine(sprintf "Values: %s" (benchmark'.ToMathematicaString()))

        stream.Flush()
        stream.Close()

        0 // return an integer exit code
