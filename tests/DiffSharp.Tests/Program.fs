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

    let benchmarkver = "1.0.6"

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
        let xd = DiffSharp.AD.D x
        let xdf = DiffSharp.AD.Forward.D x
        let xdr = DiffSharp.AD.Reverse.D x
        let xv = [|2.2; 3.5; 5.1|]
        let xvd = Array.map DiffSharp.AD.D xv
        let xvdf = Array.map DiffSharp.AD.Forward.D xv
        let xvdr = Array.map DiffSharp.AD.Reverse.D xv
        let v = [|1.2; 3.4; 5.2|]
        let vd = Array.map DiffSharp.AD.D v
        let vdf = Array.map DiffSharp.AD.Forward.D v
        let vdr = Array.map DiffSharp.AD.Reverse.D v
        let u = [|1.5; 3.1; 5.4|]
        let ud = Array.map DiffSharp.AD.D u
        let udf = Array.map DiffSharp.AD.Forward.D u
        let udr = Array.map DiffSharp.AD.Reverse.D u

        printb 1 35 "original functions"
        let res_fss, dur_fss = duration noriginal (fun () -> (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_fvs, dur_fvs = duration noriginal (fun () -> (fun (x:float[]) -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_fvv, dur_fvv = duration noriginal (fun () -> (fun (x:float[]) -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        if dur_fss = 0. || dur_fvs = 0. || dur_fvv = 0. then printfn "***\n WARNING: Zero duration encountered for an original function\n***"

        printb 2 35 "diff"
        let res_diff_AD, dur_diff_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diff_ADF, dur_diff_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diff_ADR, dur_diff_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diff_SADF1, dur_diff_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SADF2, dur_diff_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SADFG, dur_diff_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SADFGH, dur_diff_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SADFN, dur_diff_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SADR1, dur_diff_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_N, dur_diff_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff_SCom, dur_diff_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff_Symbolic = DiffSharp.Symbolic.DiffOps.diff <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff_SUse, dur_diff_SUse = duration nsymbolic (fun () -> f_diff_Symbolic x)

        printb 3 35 "diff2"
        let res_diff2_AD, dur_diff2_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diff2_ADF, dur_diff2_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diff2_ADR, dur_diff2_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diff2_SADF1, dur_diff2_SADF1 = 0., 0.
        let res_diff2_SADF2, dur_diff2_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_SADFG, dur_diff2_SADFG = 0., 0.
        let res_diff2_SADFGH, dur_diff2_SADFGH = 0., 0.
        let res_diff2_SADFN, dur_diff2_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_SADR1, dur_diff2_SADR1 = 0., 0.
        let res_diff2_N, dur_diff2_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2_SCom, dur_diff2_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff2_Symbolic = DiffSharp.Symbolic.DiffOps.diff2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff2_SUse, dur_diff2_SUse = duration nsymbolic (fun () -> f_diff2_Symbolic x)

        printb 4 35 "diffn"
        let res_diffn_AD, dur_diffn_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diffn_ADF, dur_diffn_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diffn_ADR, dur_diffn_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diffn_SADF1, dur_diffn_SADF1 = 0., 0.
        let res_diffn_SADF2, dur_diffn_SADF2 = 0., 0.
        let res_diffn_SADFG, dur_diffn_SADFG = 0., 0.
        let res_diffn_SADFGH, dur_diffn_SADFGH = 0., 0.
        let res_diffn_SADFN, dur_diffn_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diffn_SADR1, dur_diffn_SADR1 = 0., 0.
        let res_diffn_N, dur_diffn_N = 0., 0.
        let res_diffn_SCom, dur_diffn_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diffn 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diffn_Symbolic = DiffSharp.Symbolic.DiffOps.diffn 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diffn_SUse, dur_diffn_SUse = duration nsymbolic (fun () -> f_diffn_Symbolic x)

        printb 5 35 "grad"
        let res_grad_AD, dur_grad_AD = duration n (fun () -> DiffSharp.AD.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_grad_ADF, dur_grad_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_grad_ADR, dur_grad_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_grad_SADF1, dur_grad_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SADF2, dur_grad_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SADFG, dur_grad_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SADFGH, dur_grad_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SADFN, dur_grad_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.grad (fun x ->(x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SADR1, dur_grad_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_N, dur_grad_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad_SCom, dur_grad_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.grad <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_grad_Symbolic = DiffSharp.Symbolic.DiffOps.grad <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_grad_SUse, dur_grad_SUse = duration nsymbolic (fun () -> f_grad_Symbolic xv)

        printb 6 35 "gradv"
        let res_gradv_AD, dur_gradv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_gradv_ADF, dur_gradv_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_gradv_ADR, dur_gradv_ADR = 0., 0.
        let res_gradv_SADF1, dur_gradv_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_SADF2, dur_gradv_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_SADFG, dur_gradv_SADFG = 0., 0.
        let res_gradv_SADFGH, dur_gradv_SADFGH = 0., 0.
        let res_gradv_SADFN, dur_gradv_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_SADR1, dur_gradv_SADR1 = 0., 0.
        let res_gradv_N, dur_gradv_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv_SCom, dur_gradv_SCom = 0., 0.
        let res_gradv_SUse, dur_gradv_SUse = 0., 0.

        printb 7 35 "hessian"
        let res_hessian_AD, dur_hessian_AD = duration n (fun () -> DiffSharp.AD.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_hessian_ADF, dur_hessian_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_hessian_ADR, dur_hessian_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_hessian_SADF1, dur_hessian_SADF1 = 0., 0.
        let res_hessian_SADF2, dur_hessian_SADF2 = 0., 0.
        let res_hessian_SADFG, dur_hessian_SADFG = 0., 0.
        let res_hessian_SADFGH, dur_hessian_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_SADFN, dur_hessian_SADFN = 0., 0.
        let res_hessian_SADR1, dur_hessian_SADR1 = 0., 0.
        let res_hessian_N, dur_hessian_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian_SCom, dur_hessian_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.hessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_hessian_Symbolic = DiffSharp.Symbolic.DiffOps.hessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_hessian_SUse, dur_hessian_SUse = duration nsymbolic (fun () -> f_hessian_Symbolic xv)

        printb 8 35 "hessianv"
        let res_hessianv_AD, dur_hessianv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.hessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_hessianv_ADF, dur_hessianv_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.hessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_hessianv_ADR, dur_hessianv_ADR = 0., 0.
        let res_hessianv_SADF1, dur_hessianv_SADF1 = 0., 0.
        let res_hessianv_SADF2, dur_hessianv_SADF2 = 0., 0.
        let res_hessianv_SADFG, dur_hessianv_SADFG = 0., 0.
        let res_hessianv_SADFGH, dur_hessianv_SADFGH = 0., 0.
        let res_hessianv_SADFN, dur_hessianv_SADFN = 0., 0.
        let res_hessianv_SADR1, dur_hessianv_SADR1 = 0., 0.
        let res_hessianv_N, dur_hessianv_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv_SCom, dur_hessianv_SCom = 0., 0.
        let res_hessianv_SUse, dur_hessianv_SUse = 0., 0.

        printb 9 35 "gradhessian"
        let res_gradhessian_AD, dur_gradhessian_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_gradhessian_ADF, dur_gradhessian_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_gradhessian_ADR, dur_gradhessian_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_gradhessian_SADF1, dur_gradhessian_SADF1 = 0., 0.
        let res_gradhessian_SADF2, dur_gradhessian_SADF2 = 0., 0.
        let res_gradhessian_SADFG, dur_gradhessian_SADFG = 0., 0.
        let res_gradhessian_SADFGH, dur_gradhessian_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_SADFN, dur_gradhessian_SADFN = 0., 0.
        let res_gradhessian_SADR1, dur_gradhessian_SADR1 = 0., 0.
        let res_gradhessian_N, dur_gradhessian_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian_SCom, dur_gradhessian_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.gradhessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_gradhessian_Symbolic = DiffSharp.Symbolic.DiffOps.gradhessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_gradhessian_SUse, dur_gradhessian_SUse = duration nsymbolic (fun () -> f_gradhessian_Symbolic xv)

        printb 10 35 "gradhessianv"
        let res_gradhessianv_AD, dur_gradhessianv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradhessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_gradhessianv_ADF, dur_gradhessianv_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradhessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_gradhessianv_ADR, dur_gradhessianv_ADR = 0., 0.
        let res_gradhessianv_SADF1, dur_gradhessianv_SADF1 = 0., 0.
        let res_gradhessianv_SADF2, dur_gradhessianv_SADF2 = 0., 0.
        let res_gradhessianv_SADFG, dur_gradhessianv_SADFG = 0., 0.
        let res_gradhessianv_SADFGH, dur_gradhessianv_SADFGH = 0., 0.
        let res_gradhessianv_SADFN, dur_gradhessianv_SADFN = 0., 0.
        let res_gradhessianv_SADR1, dur_gradhessianv_SADR1 = 0., 0.
        let res_gradhessianv_N, dur_gradhessianv_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessianv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv_SCom, dur_gradhessianv_SCom = 0., 0.
        let res_gradhessianv_SUse, dur_gradhessianv_SUse = 0., 0.

        printb 11 35 "laplacian"
        let res_laplacian_AD, dur_laplacian_AD = duration n (fun () -> DiffSharp.AD.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_laplacian_ADF, dur_laplacian_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_laplacian_ADR, dur_laplacian_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_laplacian_SADF1, dur_laplacian_SADF1 = 0., 0.
        let res_laplacian_SADF2, dur_laplacian_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_SADFG, dur_laplacian_SADFG = 0., 0.
        let res_laplacian_SADFGH, dur_laplacian_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_SADFN, dur_laplacian_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_SADR1, dur_laplacian_SADR1 = 0., 0.
        let res_laplacian_N, dur_laplacian_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian_SCom, dur_laplacian_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.laplacian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_laplacian_Symbolic = DiffSharp.Symbolic.DiffOps.laplacian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_laplacian_SUse, dur_laplacian_SUse = duration nsymbolic (fun () -> f_laplacian_Symbolic xv)

        printb 12 35 "jacobian"
        let res_jacobian_AD, dur_jacobian_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_jacobian_ADF, dur_jacobian_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_jacobian_ADR, dur_jacobian_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_jacobian_SADF1, dur_jacobian_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SADF2, dur_jacobian_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SADFG, dur_jacobian_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SADFGH, dur_jacobian_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SADFN, dur_jacobian_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SADR1, dur_jacobian_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_N, dur_jacobian_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian_SCom, dur_jacobian_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobian <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobian_Symbolic = DiffSharp.Symbolic.DiffOps.jacobian <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobian_SUse, dur_jacobian_SUse = duration nsymbolic (fun () -> f_jacobian_Symbolic xv)

        printb 13 35 "jacobianv"
        let res_jacobianv_AD, dur_jacobianv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd vd)
        let res_jacobianv_ADF, dur_jacobianv_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf vdf)
        let res_jacobianv_ADR, dur_jacobianv_ADR = 0., 0.
        let res_jacobianv_SADF1, dur_jacobianv_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_SADF2, dur_jacobianv_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_SADFG, dur_jacobianv_SADFG =  0., 0.
        let res_jacobianv_SADFGH, dur_jacobianv_SADFGH = 0., 0.
        let res_jacobianv_SADFN, dur_jacobianv_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_SADR1, dur_jacobianv_SADR1 = 0., 0.
        let res_jacobianv_N, dur_jacobianv_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv_SCom, dur_jacobianv_SCom =  0., 0.
        let res_jacobianv_SUse, dur_jacobianv_SUse =  0., 0.

        printb 14 35 "jacobianT"
        let res_jacobianT_AD, dur_jacobianT_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_jacobianT_ADF, dur_jacobianT_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_jacobianT_ADR, dur_jacobianT_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_jacobianT_SADF1, dur_jacobianT_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SADF2, dur_jacobianT_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SADFG, dur_jacobianT_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SADFGH, dur_jacobianT_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SADFN, dur_jacobianT_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SADR1, dur_jacobianT_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_N, dur_jacobianT_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT_SCom, dur_jacobianT_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobianT <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobianT_Symbolic = DiffSharp.Symbolic.DiffOps.jacobianT <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobianT_SUse, dur_jacobianT_SUse = duration nsymbolic (fun () -> f_jacobianT_Symbolic xv)

        printb 15 35 "jacobianTv"
        let res_jacobianTv_AD, dur_jacobianTv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd ud)
        let res_jacobianTv_ADF, dur_jacobianTv_ADF = 0., 0.
        let res_jacobianTv_ADR, dur_jacobianTv_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr udr)
        let res_jacobianTv_SADF1, dur_jacobianTv_SADF1 = 0., 0.
        let res_jacobianTv_SADF2, dur_jacobianTv_SADF2 = 0., 0.
        let res_jacobianTv_SADFG, dur_jacobianTv_SADFG =  0., 0.
        let res_jacobianTv_SADFGH, dur_jacobianTv_SADFGH = 0., 0.
        let res_jacobianTv_SADFN, dur_jacobianTv_SADFN = 0., 0.
        let res_jacobianTv_SADR1, dur_jacobianTv_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv_N, dur_jacobianTv_N = 0., 0.
        let res_jacobianTv_SCom, dur_jacobianTv_SCom = 0., 0.
        let res_jacobianTv_SUse, dur_jacobianTv_SUse = 0., 0.

        printb 16 35 "curl"
        let res_curl_AD, dur_curl_AD = duration n (fun () -> DiffSharp.AD.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_curl_ADF, dur_curl_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_curl_ADR, dur_curl_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_curl_SADF1, dur_curl_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SADF2, dur_curl_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SADFG, dur_curl_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SADFGH, dur_curl_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SADFN, dur_curl_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SADR1, dur_curl_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_N, dur_curl_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.curl (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl_SCom, dur_curl_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curl <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curl_Symbolic = DiffSharp.Symbolic.DiffOps.curl <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curl_SUse, dur_curl_SUse = duration nsymbolic (fun () -> f_curl_Symbolic xv)

        printb 17 35 "div"
        let res_div_AD, dur_div_AD = duration n (fun () -> DiffSharp.AD.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_div_ADF, dur_div_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_div_ADR, dur_div_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_div_SADF1, dur_div_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SADF2, dur_div_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SADFG, dur_div_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SADFGH, dur_div_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SADFN, dur_div_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SADR1, dur_div_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_N, dur_div_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.div (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div_SCom, dur_div_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.div <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_div_Symbolic = DiffSharp.Symbolic.DiffOps.div <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_div_SUse, dur_div_SUse = duration nsymbolic (fun () -> f_div_Symbolic xv)
        
        printb 18 35 "curldiv"
        let res_curldiv_AD, dur_curldiv_AD = duration n (fun () -> DiffSharp.AD.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_curldiv_ADF, dur_curldiv_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_curldiv_ADR, dur_curldiv_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_curldiv_SADF1, dur_curldiv_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SADF2, dur_curldiv_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SADFG, dur_curldiv_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SADFGH, dur_curldiv_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SADFN, dur_curldiv_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SADR1, dur_curldiv_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_N, dur_curldiv_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.curldiv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv_SCom, dur_curldiv_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curldiv <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curldiv_Symbolic = DiffSharp.Symbolic.DiffOps.curldiv <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curldiv_SUse, dur_curldiv_SUse = duration nsymbolic (fun () -> f_curldiv_Symbolic xv)

        //
        //
        //
        //
        //

        printb 19 35 "diff'"
        let res_diff'_AD, dur_diff'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diff'_ADF, dur_diff'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diff'_ADR, dur_diff'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diff'_SADF1, dur_diff'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SADF2, dur_diff'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SADFG, dur_diff'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SADFGH, dur_diff'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SADFN, dur_diff'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SADR1, dur_diff'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_N, dur_diff'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff'_SCom, dur_diff'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff'_Symbolic = DiffSharp.Symbolic.DiffOps.diff' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff'_SUse, dur_diff'_SUse = duration nsymbolic (fun () -> f_diff'_Symbolic x)

        printb 20 35 "diff2'"
        let res_diff2'_AD, dur_diff2'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diff2'_ADF, dur_diff2'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diff2'_ADR, dur_diff2'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diff2'_SADF1, dur_diff2'_SADF1 = 0., 0.
        let res_diff2'_SADF2, dur_diff2'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_SADFG, dur_diff2'_SADFG = 0., 0.
        let res_diff2'_SADFGH, dur_diff2'_SADFGH = 0., 0.
        let res_diff2'_SADFN, dur_diff2'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_SADR1, dur_diff2'_SADR1 = 0., 0.
        let res_diff2'_N, dur_diff2'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diff2'_SCom, dur_diff2'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diff2' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diff2'_Symbolic = DiffSharp.Symbolic.DiffOps.diff2' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diff2'_SUse, dur_diff2'_SUse = duration nsymbolic (fun () -> f_diff2'_Symbolic x)

        printb 21 35 "diffn'"
        let res_diffn'_AD, dur_diffn'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xd)
        let res_diffn'_ADF, dur_diffn'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdf)
        let res_diffn'_ADR, dur_diffn'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) xdr)
        let res_diffn'_SADF1, dur_diffn'_SADF1 = 0., 0.
        let res_diffn'_SADF2, dur_diffn'_SADF2 = 0., 0.
        let res_diffn'_SADFG, dur_diffn'_SADFG = 0., 0.
        let res_diffn'_SADFGH, dur_diffn'_SADFGH = 0., 0.
        let res_diffn'_SADFN, dur_diffn'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
        let res_diffn'_SADR1, dur_diffn'_SADR1 = 0., 0.
        let res_diffn'_N, dur_diffn'_N = 0., 0.
        let res_diffn'_SCom, dur_diffn'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.diffn' 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>)
        let f_diffn'_Symbolic = DiffSharp.Symbolic.DiffOps.diffn' 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @>
        let res_diffn'_SUse, dur_diffn'_SUse = duration nsymbolic (fun () -> f_diffn'_Symbolic x)

        printb 22 35 "grad'"
        let res_grad'_AD, dur_grad'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_grad'_ADF, dur_grad'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_grad'_ADR, dur_grad'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_grad'_SADF1, dur_grad'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SADF2, dur_grad'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SADFG, dur_grad'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SADFGH, dur_grad'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SADFN, dur_grad'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SADR1, dur_grad'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_N, dur_grad'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_grad'_SCom, dur_grad'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.grad' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_grad'_Symbolic = DiffSharp.Symbolic.DiffOps.grad' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_grad'_SUse, dur_grad'_SUse = duration nsymbolic (fun () -> f_grad'_Symbolic xv)

        printb 23 35 "gradv'"
        let res_gradv'_AD, dur_gradv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_gradv'_ADF, dur_gradv'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_gradv'_ADR, dur_gradv'_ADR = 0., 0.
        let res_gradv'_SADF1, dur_gradv'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_SADF2, dur_gradv'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_SADFG, dur_gradv'_SADFG = 0., 0.
        let res_gradv'_SADFGH, dur_gradv'_SADFGH = 0., 0.
        let res_gradv'_SADFN, dur_gradv'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_SADR1, dur_gradv'_SADR1 = 0., 0.
        let res_gradv'_N, dur_gradv'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradv'_SCom, dur_gradv'_SCom = 0., 0.
        let res_gradv'_SUse, dur_gradv'_SUse = 0., 0.

        printb 24 35 "hessian'"
        let res_hessian'_AD, dur_hessian'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_hessian'_ADF, dur_hessian'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_hessian'_ADR, dur_hessian'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_hessian'_SADF1, dur_hessian'_SADF1 = 0., 0.
        let res_hessian'_SADF2, dur_hessian'_SADF2 = 0., 0.
        let res_hessian'_SADFG, dur_hessian'_SADFG = 0., 0.
        let res_hessian'_SADFGH, dur_hessian'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_SADFN, dur_hessian'_SADFN = 0., 0.
        let res_hessian'_SADR1, dur_hessian'_SADR1 = 0., 0.
        let res_hessian'_N, dur_hessian'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_hessian'_SCom, dur_hessian'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.hessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_hessian'_Symbolic = DiffSharp.Symbolic.DiffOps.hessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_hessian'_SUse, dur_hessian'_SUse = duration nsymbolic (fun () -> f_hessian'_Symbolic xv)

        printb 25 35 "hessianv'"
        let res_hessianv'_AD, dur_hessianv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.hessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_hessianv'_ADF, dur_hessianv'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.hessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_hessianv'_ADR, dur_hessianv'_ADR = 0., 0.
        let res_hessianv'_SADF1, dur_hessianv'_SADF1 = 0., 0.
        let res_hessianv'_SADF2, dur_hessianv'_SADF2 = 0., 0.
        let res_hessianv'_SADFG, dur_hessianv'_SADFG = 0., 0.
        let res_hessianv'_SADFGH, dur_hessianv'_SADFGH = 0., 0.
        let res_hessianv'_SADFN, dur_hessianv'_SADFN = 0., 0.
        let res_hessianv'_SADR1, dur_hessianv'_SADR1 = 0., 0.
        let res_hessianv'_N, dur_hessianv'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.hessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_hessianv'_SCom, dur_hessianv'_SCom = 0., 0.
        let res_hessianv'_SUse, dur_hessianv'_SUse = 0., 0.

        printb 26 35 "gradhessian'"
        let res_gradhessian'_AD, dur_gradhessian'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_gradhessian'_ADF, dur_gradhessian'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_gradhessian'_ADR, dur_gradhessian'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_gradhessian'_SADF1, dur_gradhessian'_SADF1 = 0., 0.
        let res_gradhessian'_SADF2, dur_gradhessian'_SADF2 = 0., 0.
        let res_gradhessian'_SADFG, dur_gradhessian'_SADFG = 0., 0.
        let res_gradhessian'_SADFGH, dur_gradhessian'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_SADFN, dur_gradhessian'_SADFN = 0., 0.
        let res_gradhessian'_SADR1, dur_gradhessian'_SADR1 = 0., 0.
        let res_gradhessian'_N, dur_gradhessian'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_gradhessian'_SCom, dur_gradhessian'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.gradhessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_gradhessian'_Symbolic = DiffSharp.Symbolic.DiffOps.gradhessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_gradhessian'_SUse, dur_gradhessian'_SUse = duration nsymbolic (fun () -> f_gradhessian'_Symbolic xv)

        printb 27 35 "gradhessianv'"
        let res_gradhessianv'_AD, dur_gradhessianv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.gradhessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd vd)
        let res_gradhessianv'_ADF, dur_gradhessianv'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.gradhessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf vdf)
        let res_gradhessianv'_ADR, dur_gradhessianv'_ADR = 0., 0.
        let res_gradhessianv'_SADF1, dur_gradhessianv'_SADF1 = 0., 0.
        let res_gradhessianv'_SADF2, dur_gradhessianv'_SADF2 = 0., 0.
        let res_gradhessianv'_SADFG, dur_gradhessianv'_SADFG = 0., 0.
        let res_gradhessianv'_SADFGH, dur_gradhessianv'_SADFGH = 0., 0.
        let res_gradhessianv'_SADFN, dur_gradhessianv'_SADFN = 0., 0.
        let res_gradhessianv'_SADR1, dur_gradhessianv'_SADR1 = 0., 0.
        let res_gradhessianv'_N, dur_gradhessianv'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.gradhessianv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
        let res_gradhessianv'_SCom, dur_gradhessianv'_SCom = 0., 0.
        let res_gradhessianv'_SUse, dur_gradhessianv'_SUse = 0., 0.

        printb 28 35 "laplacian'"
        let res_laplacian'_AD, dur_laplacian'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvd)
        let res_laplacian'_ADF, dur_laplacian'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdf)
        let res_laplacian'_ADR, dur_laplacian'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xvdr)
        let res_laplacian'_SADF1, dur_laplacian'_SADF1 = 0., 0.
        let res_laplacian'_SADF2, dur_laplacian'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_SADFG, dur_laplacian'_SADFG = 0., 0.
        let res_laplacian'_SADFGH, dur_laplacian'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_SADFN, dur_laplacian'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_SADR1, dur_laplacian'_SADR1 = 0., 0.
        let res_laplacian'_N, dur_laplacian'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
        let res_laplacian'_SCom, dur_laplacian'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.laplacian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>)
        let f_laplacian'_Symbolic = DiffSharp.Symbolic.DiffOps.laplacian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @>
        let res_laplacian'_SUse, dur_laplacian'_SUse = duration nsymbolic (fun () -> f_laplacian'_Symbolic xv)

        printb 29 35 "jacobian'"
        let res_jacobian'_AD, dur_jacobian'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_jacobian'_ADF, dur_jacobian'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_jacobian'_ADR, dur_jacobian'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_jacobian'_SADF1, dur_jacobian'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SADF2, dur_jacobian'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SADFG, dur_jacobian'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SADFGH, dur_jacobian'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SADFN, dur_jacobian'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SADR1, dur_jacobian'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_N, dur_jacobian'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobian'_SCom, dur_jacobian'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobian' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobian'_Symbolic = DiffSharp.Symbolic.DiffOps.jacobian' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobian'_SUse, dur_jacobian'_SUse = duration nsymbolic (fun () -> f_jacobian'_Symbolic xv)

        printb 30 35 "jacobianv'"
        let res_jacobianv'_AD, dur_jacobianv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd vd)
        let res_jacobianv'_ADF, dur_jacobianv'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf vdf)
        let res_jacobianv'_ADR, dur_jacobianv'_ADR = 0., 0.
        let res_jacobianv'_SADF1, dur_jacobianv'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_SADF2, dur_jacobianv'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_SADFG, dur_jacobianv'_SADFG = 0., 0.
        let res_jacobianv'_SADFGH, dur_jacobianv'_SADFGH = 0., 0.
        let res_jacobianv'_SADFN, dur_jacobianv'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_SADR1, dur_jacobianv'_SADR1 = 0., 0.
        let res_jacobianv'_N, dur_jacobianv'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
        let res_jacobianv'_SCom, dur_jacobianv'_SCom = 0., 0.
        let res_jacobianv'_SUse, dur_jacobianv'_SUse = 0., 0.

        printb 31 35 "jacobianT'"
        let res_jacobianT'_AD, dur_jacobianT'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_jacobianT'_ADF, dur_jacobianT'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_jacobianT'_ADR, dur_jacobianT'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_jacobianT'_SADF1, dur_jacobianT'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SADF2, dur_jacobianT'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SADFG, dur_jacobianT'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SADFGH, dur_jacobianT'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SADFN, dur_jacobianT'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SADR1, dur_jacobianT'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_N, dur_jacobianT'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_jacobianT'_SCom, dur_jacobianT'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.jacobianT' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_jacobianT'_Symbolic = DiffSharp.Symbolic.DiffOps.jacobianT' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_jacobianT'_SUse, dur_jacobianT'_SUse = duration nsymbolic (fun () -> f_jacobianT'_Symbolic xv)

        printb 32 35 "jacobianTv'"
        let res_jacobianTv'_AD, dur_jacobianTv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd ud)
        let res_jacobianTv'_ADF, dur_jacobianTv'_ADF = 0., 0.
        let res_jacobianTv'_ADR, dur_jacobianTv'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr udr)
        let res_jacobianTv'_SADF1, dur_jacobianTv'_SADF1 = 0., 0.
        let res_jacobianTv'_SADF2, dur_jacobianTv'_SADF2 = 0., 0.
        let res_jacobianTv'_SADFG, dur_jacobianTv'_SADFG = 0., 0.
        let res_jacobianTv'_SADFGH, dur_jacobianTv'_SADFGH = 0., 0.
        let res_jacobianTv'_SADFN, dur_jacobianTv'_SADFN = 0., 0.
        let res_jacobianTv'_SADR1, dur_jacobianTv'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
        let res_jacobianTv'_N, dur_jacobianTv'_N = 0., 0.
        let res_jacobianTv'_SCom, dur_jacobianTv'_SCom = 0., 0.
        let res_jacobianTv'_SUse, dur_jacobianTv'_SUse = 0., 0.

        printb 33 35 "curl'"
        let res_curl'_AD, dur_curl'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_curl'_ADF, dur_curl'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_curl'_ADR, dur_curl'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_curl'_SADF1, dur_curl'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SADF2, dur_curl'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SADFG, dur_curl'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SADFGH, dur_curl'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SADFN, dur_curl'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SADR1, dur_curl'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_N, dur_curl'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.curl' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curl'_SCom, dur_curl'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curl' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curl'_Symbolic = DiffSharp.Symbolic.DiffOps.curl' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curl'_SUse, dur_curl'_SUse = duration nsymbolic (fun () -> f_curl'_Symbolic xv)

        printb 34 35 "div'"
        let res_div'_AD, dur_div'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_div'_ADF, dur_div'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_div'_ADR, dur_div'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_div'_SADF1, dur_div'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SADF2, dur_div'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SADFG, dur_div'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SADFGH, dur_div'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SADFN, dur_div'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SADR1, dur_div'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_N, dur_div'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.div' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_div'_SCom, dur_div'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.div' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_div'_Symbolic = DiffSharp.Symbolic.DiffOps.div' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_div'_SUse, dur_div'_SUse = duration nsymbolic (fun () -> f_div'_Symbolic xv)

        printb 35 35 "curldiv'"
        let res_curldiv'_AD, dur_curldiv'_AD = duration n (fun () -> DiffSharp.AD.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvd)
        let res_curldiv'_ADF, dur_curldiv'_ADF = duration n (fun () -> DiffSharp.AD.Forward.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdf)
        let res_curldiv'_ADR, dur_curldiv'_ADR = duration n (fun () -> DiffSharp.AD.Reverse.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xvdr)
        let res_curldiv'_SADF1, dur_curldiv'_SADF1 = duration n (fun () -> DiffSharp.AD.Specialized.Forward1.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SADF2, dur_curldiv'_SADF2 = duration n (fun () -> DiffSharp.AD.Specialized.Forward2.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SADFG, dur_curldiv'_SADFG = duration n (fun () -> DiffSharp.AD.Specialized.ForwardG.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SADFGH, dur_curldiv'_SADFGH = duration n (fun () -> DiffSharp.AD.Specialized.ForwardGH.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SADFN, dur_curldiv'_SADFN = duration n (fun () -> DiffSharp.AD.Specialized.ForwardN.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SADR1, dur_curldiv'_SADR1 = duration n (fun () -> DiffSharp.AD.Specialized.Reverse1.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_N, dur_curldiv'_N = duration n (fun () -> DiffSharp.Numerical.DiffOps.curldiv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
        let res_curldiv'_SCom, dur_curldiv'_SCom = duration nsymbolic (fun () -> DiffSharp.Symbolic.DiffOps.curldiv' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>)
        let f_curldiv'_Symbolic = DiffSharp.Symbolic.DiffOps.curldiv' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @>
        let res_curldiv'_SUse, dur_curldiv'_SUse = duration nsymbolic (fun () -> f_curldiv'_Symbolic xv)

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

        let row_originals = vector [dur_fss; dur_fss; dur_fss; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv; dur_fvv]
        let row_AD = vector [dur_diff_AD; dur_diff2_AD; dur_diffn_AD; dur_grad_AD; dur_gradv_AD; dur_hessian_AD; dur_hessianv_AD; dur_gradhessian_AD; dur_gradhessianv_AD; dur_laplacian_AD; dur_jacobian_AD; dur_jacobianv_AD; dur_jacobianT_AD; dur_jacobianTv_AD; dur_curl_AD; dur_div_AD; dur_curldiv_AD]
        let row_ADF = vector [dur_diff_ADF; dur_diff2_ADF; dur_diffn_ADF; dur_grad_ADF; dur_gradv_ADF; dur_hessian_ADF; dur_hessianv_ADF; dur_gradhessian_ADF; dur_gradhessianv_ADF; dur_laplacian_ADF; dur_jacobian_ADF; dur_jacobianv_ADF; dur_jacobianT_ADF; dur_jacobianTv_ADF; dur_curl_ADF; dur_div_ADF; dur_curldiv_ADF]
        let row_ADR = vector [dur_diff_ADR; dur_diff2_ADR; dur_diffn_ADR; dur_grad_ADR; dur_gradv_ADR; dur_hessian_ADR; dur_hessianv_ADR; dur_gradhessian_ADR; dur_gradhessianv_ADR; dur_laplacian_ADR; dur_jacobian_ADR; dur_jacobianv_ADR; dur_jacobianT_ADR; dur_jacobianTv_ADR; dur_curl_ADR; dur_div_ADR; dur_curldiv_ADR]
        let row_SADF1 = vector [dur_diff_SADF1; dur_diff2_SADF1; dur_diffn_SADF1; dur_grad_SADF1; dur_gradv_SADF1; dur_hessian_SADF1; dur_hessianv_SADF1; dur_gradhessian_SADF1; dur_gradhessianv_SADF1; dur_laplacian_SADF1; dur_jacobian_SADF1; dur_jacobianv_SADF1; dur_jacobianT_SADF1; dur_jacobianTv_SADF1; dur_curl_SADF1; dur_div_SADF1; dur_curldiv_SADF1]
        let row_SADF2 = vector [dur_diff_SADF2; dur_diff2_SADF2; dur_diffn_SADF2; dur_grad_SADF2; dur_gradv_SADF2; dur_hessian_SADF2; dur_hessianv_SADF2; dur_gradhessian_SADF2; dur_gradhessianv_SADF2; dur_laplacian_SADF2; dur_jacobian_SADF2; dur_jacobianv_SADF2; dur_jacobianT_SADF2; dur_jacobianTv_SADF2; dur_curl_SADF2; dur_div_SADF2; dur_curldiv_SADF2]
        let row_SADFG = vector [dur_diff_SADFG; dur_diff2_SADFG; dur_diffn_SADFG; dur_grad_SADFG; dur_gradv_SADFG; dur_hessian_SADFG; dur_hessianv_SADFG; dur_gradhessian_SADFG; dur_gradhessianv_SADFG; dur_laplacian_SADFG; dur_jacobian_SADFG; dur_jacobianv_SADFG; dur_jacobianT_SADFG; dur_jacobianTv_SADFG; dur_curl_SADFG; dur_div_SADFG; dur_curldiv_SADFG]
        let row_SADFGH = vector [dur_diff_SADFGH; dur_diff2_SADFGH; dur_diffn_SADFGH; dur_grad_SADFGH; dur_gradv_SADFGH; dur_hessian_SADFGH; dur_hessianv_SADFGH; dur_gradhessian_SADFGH; dur_gradhessianv_SADFGH; dur_laplacian_SADFGH; dur_jacobian_SADFGH; dur_jacobianv_SADFGH; dur_jacobianT_SADFGH; dur_jacobianTv_SADFGH; dur_curl_SADFGH; dur_div_SADFGH; dur_curldiv_SADFGH]
        let row_SADFN = vector [dur_diff_SADFN; dur_diff2_SADFN; dur_diffn_SADFN; dur_grad_SADFN; dur_gradv_SADFN; dur_hessian_SADFN; dur_hessianv_SADFN; dur_gradhessian_SADFN; dur_gradhessianv_SADFN; dur_laplacian_SADFN; dur_jacobian_SADFN; dur_jacobianv_SADFN; dur_jacobianT_SADFN; dur_jacobianTv_SADFN; dur_curl_SADFN; dur_div_SADFN; dur_curldiv_SADFN]
        let row_SADR1 = vector [dur_diff_SADR1; dur_diff2_SADR1; dur_diffn_SADR1; dur_grad_SADR1; dur_gradv_SADR1; dur_hessian_SADR1; dur_hessianv_SADR1; dur_gradhessian_SADR1; dur_gradhessianv_SADR1; dur_laplacian_SADR1; dur_jacobian_SADR1; dur_jacobianv_SADR1; dur_jacobianT_SADR1; dur_jacobianTv_SADR1; dur_curl_SADR1; dur_div_SADR1; dur_curldiv_SADR1]
        let row_N = vector [dur_diff_N; dur_diff2_N; dur_diffn_N; dur_grad_N; dur_gradv_N; dur_hessian_N; dur_hessianv_N; dur_gradhessian_N; dur_gradhessianv_N; dur_laplacian_N; dur_jacobian_N; dur_jacobianv_N; dur_jacobianT_N; dur_jacobianTv_N; dur_curl_N; dur_div_N; dur_curldiv_N]
        let row_SCom = vector [dur_diff_SCom; dur_diff2_SCom; dur_diffn_SCom; dur_grad_SCom; dur_gradv_SCom; dur_hessian_SCom; dur_hessianv_SCom; dur_gradhessian_SCom; dur_gradhessianv_SCom; dur_laplacian_SCom; dur_jacobian_SCom; dur_jacobianv_SCom; dur_jacobianT_SCom; dur_jacobianTv_SCom; dur_curl_SCom; dur_div_SCom; dur_curldiv_SCom]
        let row_SUse = vector [dur_diff_SUse; dur_diff2_SUse; dur_diffn_SUse; dur_grad_SUse; dur_gradv_SUse; dur_hessian_SUse; dur_hessianv_SUse; dur_gradhessian_SUse; dur_gradhessianv_SUse; dur_laplacian_SUse; dur_jacobian_SUse; dur_jacobianv_SUse; dur_jacobianT_SUse; dur_jacobianTv_SUse; dur_curl_SUse; dur_div_SUse; dur_curldiv_SUse]

        let benchmark = matrix [Vector.toSeq (row_AD ./ row_originals)
                                Vector.toSeq (row_ADF ./ row_originals)
                                Vector.toSeq (row_ADR ./ row_originals)
                                Vector.toSeq (row_SADF1 ./ row_originals)
                                Vector.toSeq (row_SADF2 ./ row_originals)
                                Vector.toSeq (row_SADFG ./ row_originals)
                                Vector.toSeq (row_SADFGH ./ row_originals)
                                Vector.toSeq (row_SADFN ./ row_originals)
                                Vector.toSeq (row_SADR1 ./ row_originals)
                                Vector.toSeq (row_N ./ row_originals)
                                Vector.toSeq (row_SCom ./ row_originals)
                                Vector.toSeq (row_SUse ./ row_originals)]

        let row_AD' = vector [dur_diff'_AD; dur_diff2'_AD; dur_diffn'_AD; dur_grad'_AD; dur_gradv'_AD; dur_hessian'_AD; dur_hessianv'_AD; dur_gradhessian'_AD; dur_gradhessianv'_AD; dur_laplacian'_AD; dur_jacobian'_AD; dur_jacobianv'_AD; dur_jacobianT'_AD; dur_jacobianTv'_AD; dur_curl'_AD; dur_div'_AD; dur_curldiv'_AD]
        let row_ADF' = vector [dur_diff'_ADF; dur_diff2'_ADF; dur_diffn'_ADF; dur_grad'_ADF; dur_gradv'_ADF; dur_hessian'_ADF; dur_hessianv'_ADF; dur_gradhessian'_ADF; dur_gradhessianv'_ADF; dur_laplacian'_ADF; dur_jacobian'_ADF; dur_jacobianv'_ADF; dur_jacobianT'_ADF; dur_jacobianTv'_ADF; dur_curl'_ADF; dur_div'_ADF; dur_curldiv'_ADF]
        let row_ADR' = vector [dur_diff'_ADR; dur_diff2'_ADR; dur_diffn'_ADR; dur_grad'_ADR; dur_gradv'_ADR; dur_hessian'_ADR; dur_hessianv'_ADR; dur_gradhessian'_ADR; dur_gradhessianv'_ADR; dur_laplacian'_ADR; dur_jacobian'_ADR; dur_jacobianv'_ADR; dur_jacobianT'_ADR; dur_jacobianTv'_ADR; dur_curl'_ADR; dur_div'_ADR; dur_curldiv'_ADR]
        let row_SADF1' = vector [dur_diff'_SADF1; dur_diff2'_SADF1; dur_diffn'_SADF1; dur_grad'_SADF1; dur_gradv'_SADF1; dur_hessian'_SADF1; dur_hessianv'_SADF1; dur_gradhessian'_SADF1; dur_gradhessianv'_SADF1; dur_laplacian'_SADF1; dur_jacobian'_SADF1; dur_jacobianv'_SADF1; dur_jacobianT'_SADF1; dur_jacobianTv'_SADF1; dur_curl'_SADF1; dur_div'_SADF1; dur_curldiv'_SADF1]
        let row_SADF2' = vector [dur_diff'_SADF2; dur_diff2'_SADF2; dur_diffn'_SADF2; dur_grad'_SADF2; dur_gradv'_SADF2; dur_hessian'_SADF2; dur_hessianv'_SADF2; dur_gradhessian'_SADF2; dur_gradhessianv'_SADF2; dur_laplacian'_SADF2; dur_jacobian'_SADF2; dur_jacobianv'_SADF2; dur_jacobianT'_SADF2; dur_jacobianTv'_SADF2; dur_curl'_SADF2; dur_div'_SADF2; dur_curldiv'_SADF2]
        let row_SADFG' = vector [dur_diff'_SADFG; dur_diff2'_SADFG; dur_diffn'_SADFG; dur_grad'_SADFG; dur_gradv'_SADFG; dur_hessian'_SADFG; dur_hessianv'_SADFG; dur_gradhessian'_SADFG; dur_gradhessianv'_SADFG; dur_laplacian'_SADFG; dur_jacobian'_SADFG; dur_jacobianv'_SADFG; dur_jacobianT'_SADFG; dur_jacobianTv'_SADFG; dur_curl'_SADFG; dur_div'_SADFG; dur_curldiv'_SADFG]
        let row_SADFGH' = vector [dur_diff'_SADFGH; dur_diff2'_SADFGH; dur_diffn'_SADFGH; dur_grad'_SADFGH; dur_gradv'_SADFGH; dur_hessian'_SADFGH; dur_hessianv'_SADFGH; dur_gradhessian'_SADFGH; dur_gradhessianv'_SADFGH; dur_laplacian'_SADFGH; dur_jacobian'_SADFGH; dur_jacobianv'_SADFGH; dur_jacobianT'_SADFGH; dur_jacobianTv'_SADFGH; dur_curl'_SADFGH; dur_div'_SADFGH; dur_curldiv'_SADFGH]
        let row_SADFN' = vector [dur_diff'_SADFN; dur_diff2'_SADFN; dur_diffn'_SADFN; dur_grad'_SADFN; dur_gradv'_SADFN; dur_hessian'_SADFN; dur_hessianv'_SADFN; dur_gradhessian'_SADFN; dur_gradhessianv'_SADFN; dur_laplacian'_SADFN; dur_jacobian'_SADFN; dur_jacobianv'_SADFN; dur_jacobianT'_SADFN; dur_jacobianTv'_SADFN; dur_curl'_SADFN; dur_div'_SADFN; dur_curldiv'_SADFN]
        let row_SADR1' = vector [dur_diff'_SADR1; dur_diff2'_SADR1; dur_diffn'_SADR1; dur_grad'_SADR1; dur_gradv'_SADR1; dur_hessian'_SADR1; dur_hessianv'_SADR1; dur_gradhessian'_SADR1; dur_gradhessianv'_SADR1; dur_laplacian'_SADR1; dur_jacobian'_SADR1; dur_jacobianv'_SADR1; dur_jacobianT'_SADR1; dur_jacobianTv'_SADR1; dur_curl'_SADR1; dur_div'_SADR1; dur_curldiv'_SADR1]
        let row_N' = vector [dur_diff'_N; dur_diff2'_N; dur_diffn'_N; dur_grad'_N; dur_gradv'_N; dur_hessian'_N; dur_hessianv'_N; dur_gradhessian'_N; dur_gradhessianv'_N; dur_laplacian'_N; dur_jacobian'_N; dur_jacobianv'_N; dur_jacobianT'_N; dur_jacobianTv'_N; dur_curl'_N; dur_div'_N; dur_curldiv'_N]
        let row_SCom' = vector [dur_diff'_SCom; dur_diff2'_SCom; dur_diffn'_SCom; dur_grad'_SCom; dur_gradv'_SCom; dur_hessian'_SCom; dur_hessianv'_SCom; dur_gradhessian'_SCom; dur_gradhessianv'_SCom; dur_laplacian'_SCom; dur_jacobian'_SCom; dur_jacobianv'_SCom; dur_jacobianT'_SCom; dur_jacobianTv'_SCom; dur_curl'_SCom; dur_div'_SCom; dur_curldiv'_SCom]
        let row_SUse' = vector [dur_diff'_SUse; dur_diff2'_SUse; dur_diffn'_SUse; dur_grad'_SUse; dur_gradv'_SUse; dur_hessian'_SUse; dur_hessianv'_SUse; dur_gradhessian'_SUse; dur_gradhessianv'_SUse; dur_laplacian'_SUse; dur_jacobian'_SUse; dur_jacobianv'_SUse; dur_jacobianT'_SUse; dur_jacobianTv'_SUse; dur_curl'_SUse; dur_div'_SUse; dur_curldiv'_SUse]

        let benchmark' = matrix [Vector.toSeq (row_AD' ./ row_originals)
                                 Vector.toSeq (row_ADF' ./ row_originals)
                                 Vector.toSeq (row_ADR' ./ row_originals)
                                 Vector.toSeq (row_SADF1' ./ row_originals)
                                 Vector.toSeq (row_SADF2' ./ row_originals)
                                 Vector.toSeq (row_SADFG' ./ row_originals)
                                 Vector.toSeq (row_SADFGH' ./ row_originals)
                                 Vector.toSeq (row_SADFN' ./ row_originals)
                                 Vector.toSeq (row_SADR1' ./ row_originals)
                                 Vector.toSeq (row_N' ./ row_originals)
                                 Vector.toSeq (row_SCom' ./ row_originals)
                                 Vector.toSeq (row_SUse' ./ row_originals)]

        let score = (Vector.sum row_AD)
                    + (Vector.sum row_ADF)
                    + (Vector.sum row_ADR)
                    + (Vector.sum row_SADF1) 
                    + (Vector.sum row_SADF2)
                    + (Vector.sum row_SADFG)
                    + (Vector.sum row_SADFGH)
                    + (Vector.sum row_SADFN)
                    + (Vector.sum row_SADR1)
                    + (Vector.sum row_N)
                    + (Vector.sum row_SCom)
                    + (Vector.sum row_SUse)
    
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
        stream.WriteLine("Column labels: {diff, diff2, diffn, grad, gradv, hessian, hessianv, gradhessian, gradhessianv, laplacian, jacobian, jacobianv, jacobianT, jacobianTv, curl, div, curldiv}")
        stream.WriteLine("Row labels: {DiffSharp.AD, DiffSharp.AD.Forward, DiffSharp.AD.Reverse, DiffSharp.AD.Specialized.Forward1, DiffSharp.AD.Specialized.Forward2, DiffSharp.AD.Specialized.ForwardG, DiffSharp.AD.Specialized.ForwardGH, DiffSharp.AD.Specialized.ForwardN, DiffSharp.AD.Specialized.Reverse1, DiffSharp.Numerical, DiffSharp.Symbolic (Compile), DiffSharp.Symbolic (Use)}")
        stream.WriteLine(sprintf "Values: %s" (benchmark.ToMathematicaString()))

        stream.WriteLine("\r\nBenchmark matrix B\r\n")
        stream.WriteLine("Column labels: {diff', diff2', diffn', grad', gradv', hessian', hessianv', gradhessian', gradhessianv', laplacian', jacobian', jacobianv', jacobianT', jacobianTv', curl', div', curldiv'}")
        stream.WriteLine("Row labels: {DiffSharp.AD, DiffSharp.AD.Forward, DiffSharp.AD.Reverse, DiffSharp.AD.Specialized.Forward1, DiffSharp.AD.Specialized.Forward2, DiffSharp.AD.Specialized.ForwardG, DiffSharp.AD.Specialized.ForwardGH, DiffSharp.AD.Specialized.ForwardN, DiffSharp.AD.Specialized.Reverse1, DiffSharp.Numerical, DiffSharp.Symbolic (Compile), DiffSharp.Symbolic (Use)}")
        stream.WriteLine(sprintf "Values: %s" (benchmark'.ToMathematicaString()))

        stream.Flush()
        stream.Close()

        0 // return an integer exit code
