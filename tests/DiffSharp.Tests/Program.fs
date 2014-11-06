//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
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
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

open DiffSharp.Util.LinearAlgebra

let duration n f =
    let before = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
    for i in 1..n do // n > 100000 seems to work fine
        f() |> ignore
    let after = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.Ticks
    f(), (float (after - before)) / (float n)

[<EntryPoint>]
let main argv = 

    let benchmarkver = "1.0.0"
    let n = 100000 // n > 100000 seems to work fine
    let nsymbolic = n / 1000
    let noriginal = n * 100
    let file = sprintf "DiffSharpBenchmark%A.txt" System.DateTime.Now.Ticks

    let os = System.Environment.OSVersion.ToString()
    let clr = System.Environment.Version.ToString()

    let cpu =
        try
            let mutable cpu = ""
            let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM Win32_Processor")
            for mo in mos.Get() do
                cpu <- mo.["name"].ToString()
            cpu
        with
            | _ -> "Unknown"

    let ram =
        try
            let mutable ram = ""
            let mos = new System.Management.ManagementObjectSearcher("SELECT * FROM CIM_OperatingSystem")
            for mo in mos.Get() do
                ram <- mo.["TotalVisibleMemorySize"].ToString() + " bytes"
            ram
        with
            | _ -> "Unknown"

    printfn "DiffSharp\n"
    printfn "Benchmarking module version: %s" benchmarkver

    let diffsharpver = System.Reflection.AssemblyName.GetAssemblyName("DiffSharp.dll").Version.ToString()
    printfn "DiffSharp library version: %s\n" diffsharpver

    printfn "OS: %s" os
    printfn ".NET CLR version: %s" clr
    printfn "CPU: %s" cpu
    printfn "RAM: %s\n" ram

    printfn "Repetitions: %A\n" n

    printfn "Press any key to start benchmarking..."
    System.Console.ReadKey(true) |> ignore

    let started = System.DateTime.Now
    printfn "\nBenchmarking started: %A" started

    let x = 2.8
    let xv = [|2.2; 3.5; 5.1|]
    let v = [|1.2; 3.4; 5.2|]
    let u = [|1.5; 3.1; 5.4|]

    printfn "Running benchmark: original functions"
    let res_fss, dur_fss = duration noriginal (fun () -> (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_fvs, dur_fvs = duration noriginal (fun () -> (fun (x:float[]) -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_fvv, dur_fvv = duration noriginal (fun () -> (fun (x:float[]) -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    if dur_fss = 0. || dur_fvs = 0. || dur_fvv = 0. then printfn "Zero duration encountered for an original function"

    printfn "Running benchmark: diff"
    let res_diff_AD_Forward, dur_diff_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_AD_Forward2, dur_diff_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_AD_ForwardG, dur_diff_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_AD_ForwardGH, dur_diff_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_AD_ForwardN, dur_diff_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_AD_Reverse, dur_diff_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_Numerical, dur_diff_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.diff (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff_Symbolic, dur_diff_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diff <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: diff2"
    let res_diff2_AD_Forward, dur_diff2_AD_Forward = 0., 0.
    let res_diff2_AD_Forward2, dur_diff2_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2_AD_ForwardG, dur_diff2_AD_ForwardG = 0., 0.
    let res_diff2_AD_ForwardGH, dur_diff2_AD_ForwardGH = 0., 0.
    let res_diff2_AD_ForwardN, dur_diff2_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2_AD_Reverse, dur_diff2_AD_Reverse = 0., 0.
    let res_diff2_Numerical, dur_diff2_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.diff2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2_Symbolic, dur_diff2_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diff2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: diffn"
    let res_diffn_AD_Forward, dur_diffn_AD_Forward = 0., 0.
    let res_diffn_AD_Forward2, dur_diffn_AD_Forward2 = 0., 0.
    let res_diffn_AD_ForwardG, dur_diffn_AD_ForwardG = 0., 0.
    let res_diffn_AD_ForwardGH, dur_diffn_AD_ForwardGH = 0., 0.
    let res_diffn_AD_ForwardN, dur_diffn_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diffn 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diffn_AD_Reverse, dur_diffn_AD_Reverse = 0., 0.
    let res_diffn_Numerical, dur_diffn_Numerical = 0., 0.
    let res_diffn_Symbolic, dur_diffn_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diffn 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: grad"
    let res_grad_AD_Forward, dur_grad_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_AD_Forward2, dur_grad_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_AD_ForwardG, dur_grad_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_AD_ForwardGH, dur_grad_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_AD_ForwardN, dur_grad_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.grad (fun x ->(x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_AD_Reverse, dur_grad_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_Numerical, dur_grad_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.grad (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad_Symbolic, dur_grad_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.grad <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: gradv"
    let res_gradv_AD_Forward, dur_gradv_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv_AD_Forward2, dur_gradv_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv_AD_ForwardG, dur_gradv_AD_ForwardG = 0., 0.
    let res_gradv_AD_ForwardGH, dur_gradv_AD_ForwardGH = 0., 0.
    let res_gradv_AD_ForwardN, dur_gradv_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv_AD_Reverse, dur_gradv_AD_Reverse = 0., 0.
    let res_gradv_Numerical, dur_gradv_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.gradv (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv_Symbolic, dur_gradv_Symbolic = 0., 0.

    printfn "Running benchmark: hessian"
    let res_hessian_AD_Forward, dur_hessian_AD_Forward = 0., 0.
    let res_hessian_AD_Forward2, dur_hessian_AD_Forward2 = 0., 0.
    let res_hessian_AD_ForwardG, dur_hessian_AD_ForwardG = 0., 0.
    let res_hessian_AD_ForwardGH, dur_hessian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian_AD_ForwardN, dur_hessian_AD_ForwardN = 0., 0.
    let res_hessian_AD_Reverse, dur_hessian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian_Numerical, dur_hessian_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.hessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian_Symbolic, dur_hessian_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.hessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: gradhessian"
    let res_gradhessian_AD_Forward, dur_gradhessian_AD_Forward = 0., 0.
    let res_gradhessian_AD_Forward2, dur_gradhessian_AD_Forward2 = 0., 0.
    let res_gradhessian_AD_ForwardG, dur_gradhessian_AD_ForwardG = 0., 0.
    let res_gradhessian_AD_ForwardGH, dur_gradhessian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian_AD_ForwardN, dur_gradhessian_AD_ForwardN = 0., 0.
    let res_gradhessian_AD_Reverse, dur_gradhessian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian_Numerical, dur_gradhessian_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.gradhessian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian_Symbolic, dur_gradhessian_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.gradhessian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: laplacian"
    let res_laplacian_AD_Forward, dur_laplacian_AD_Forward = 0., 0.
    let res_laplacian_AD_Forward2, dur_laplacian_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian_AD_ForwardG, dur_laplacian_AD_ForwardG = 0., 0.
    let res_laplacian_AD_ForwardGH, dur_laplacian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian_AD_ForwardN, dur_laplacian_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian_AD_Reverse, dur_laplacian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian_Numerical, dur_laplacian_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.laplacian (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian_Symbolic, dur_laplacian_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.laplacian <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: jacobian"
    let res_jacobian_AD_Forward, dur_jacobian_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_AD_Forward2, dur_jacobian_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_AD_ForwardG, dur_jacobian_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_AD_ForwardGH, dur_jacobian_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_AD_ForwardN, dur_jacobian_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_AD_Reverse, dur_jacobian_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_Numerical, dur_jacobian_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobian (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian_Symbolic, dur_jacobian_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.jacobian <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @> xv)

    printfn "Running benchmark: jacobianv"
    let res_jacobianv_AD_Forward, dur_jacobianv_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv_AD_Forward2, dur_jacobianv_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv_AD_ForwardG, dur_jacobianv_AD_ForwardG =  0., 0.
    let res_jacobianv_AD_ForwardGH, dur_jacobianv_AD_ForwardGH = 0., 0.
    let res_jacobianv_AD_ForwardN, dur_jacobianv_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv_AD_Reverse, dur_jacobianv_AD_Reverse = 0., 0.
    let res_jacobianv_Numerical, dur_jacobianv_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobianv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv_Symbolic, dur_jacobianv_Symbolic =  0., 0.

    printfn "Running benchmark: jacobianT"
    let res_jacobianT_AD_Forward, dur_jacobianT_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_AD_Forward2, dur_jacobianT_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_AD_ForwardG, dur_jacobianT_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_AD_ForwardGH, dur_jacobianT_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_AD_ForwardN, dur_jacobianT_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_AD_Reverse, dur_jacobianT_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_Numerical, dur_jacobianT_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobianT (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT_Symbolic, dur_jacobianT_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.jacobianT <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @> xv)

    printfn "Running benchmark: jacobianTv"
    let res_jacobianTv_AD_Forward, dur_jacobianTv_AD_Forward = 0., 0.
    let res_jacobianTv_AD_Forward2, dur_jacobianTv_AD_Forward2 = 0., 0.
    let res_jacobianTv_AD_ForwardG, dur_jacobianTv_AD_ForwardG =  0., 0.
    let res_jacobianTv_AD_ForwardGH, dur_jacobianTv_AD_ForwardGH = 0., 0.
    let res_jacobianTv_AD_ForwardN, dur_jacobianTv_AD_ForwardN = 0., 0.
    let res_jacobianTv_AD_Reverse, dur_jacobianTv_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobianTv (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
    let res_jacobianTv_Numerical, dur_jacobianTv_Numerical = 0., 0.
    let res_jacobianTv_Symbolic, dur_jacobianTv_Symbolic = 0., 0.

    //
    //
    //
    //
    //

    printfn "Running benchmark: diff'"
    let res_diff'_AD_Forward, dur_diff'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_AD_Forward2, dur_diff'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_AD_ForwardG, dur_diff'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_AD_ForwardGH, dur_diff'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_AD_ForwardN, dur_diff'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_AD_Reverse, dur_diff'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_Numerical, dur_diff'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.diff' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff'_Symbolic, dur_diff'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diff' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: diff2'"
    let res_diff2'_AD_Forward, dur_diff2'_AD_Forward = 0., 0.
    let res_diff2'_AD_Forward2, dur_diff2'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2'_AD_ForwardG, dur_diff2'_AD_ForwardG = 0., 0.
    let res_diff2'_AD_ForwardGH, dur_diff2'_AD_ForwardGH = 0., 0.
    let res_diff2'_AD_ForwardN, dur_diff2'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2'_AD_Reverse, dur_diff2'_AD_Reverse = 0., 0.
    let res_diff2'_Numerical, dur_diff2'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.diff2' (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diff2'_Symbolic, dur_diff2'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diff2' <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: diffn'"
    let res_diffn'_AD_Forward, dur_diffn'_AD_Forward = 0., 0.
    let res_diffn'_AD_Forward2, dur_diffn'_AD_Forward2 = 0., 0.
    let res_diffn'_AD_ForwardG, dur_diffn'_AD_ForwardG = 0., 0.
    let res_diffn'_AD_ForwardGH, dur_diffn'_AD_ForwardGH = 0., 0.
    let res_diffn'_AD_ForwardN, dur_diffn'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.diffn' 2 (fun x -> (sin (sqrt (x + 2.))) ** 3.) x)
    let res_diffn'_AD_Reverse, dur_diffn'_AD_Reverse = 0., 0.
    let res_diffn'_Numerical, dur_diffn'_Numerical = 0., 0.
    let res_diffn'_Symbolic, dur_diffn'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.diffn' 2 <@ (fun x -> (sin (sqrt (x + 2.))) ** 3.) @> x)

    printfn "Running benchmark: grad'"
    let res_grad'_AD_Forward, dur_grad'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_AD_Forward2, dur_grad'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_AD_ForwardG, dur_grad'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_AD_ForwardGH, dur_grad'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_AD_ForwardN, dur_grad'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_AD_Reverse, dur_grad'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_Numerical, dur_grad'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.grad' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_grad'_Symbolic, dur_grad'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.grad' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: gradv'"
    let res_gradv'_AD_Forward, dur_gradv'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv'_AD_Forward2, dur_gradv'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv'_AD_ForwardG, dur_gradv'_AD_ForwardG = 0., 0.
    let res_gradv'_AD_ForwardGH, dur_gradv'_AD_ForwardGH = 0., 0.
    let res_gradv'_AD_ForwardN, dur_gradv'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv'_AD_Reverse, dur_gradv'_AD_Reverse = 0., 0.
    let res_gradv'_Numerical, dur_gradv'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.gradv' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv v)
    let res_gradv'_Symbolic, dur_gradv'_Symbolic = 0., 0.

    printfn "Running benchmark: hessian'"
    let res_hessian'_AD_Forward, dur_hessian'_AD_Forward = 0., 0.
    let res_hessian'_AD_Forward2, dur_hessian'_AD_Forward2 = 0., 0.
    let res_hessian'_AD_ForwardG, dur_hessian'_AD_ForwardG = 0., 0.
    let res_hessian'_AD_ForwardGH, dur_hessian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian'_AD_ForwardN, dur_hessian'_AD_ForwardN = 0., 0.
    let res_hessian'_AD_Reverse, dur_hessian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian'_Numerical, dur_hessian'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.hessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_hessian'_Symbolic, dur_hessian'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.hessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: gradhessian'"
    let res_gradhessian'_AD_Forward, dur_gradhessian'_AD_Forward = 0., 0.
    let res_gradhessian'_AD_Forward2, dur_gradhessian'_AD_Forward2 = 0., 0.
    let res_gradhessian'_AD_ForwardG, dur_gradhessian'_AD_ForwardG = 0., 0.
    let res_gradhessian'_AD_ForwardGH, dur_gradhessian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian'_AD_ForwardN, dur_gradhessian'_AD_ForwardN = 0., 0.
    let res_gradhessian'_AD_Reverse, dur_gradhessian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian'_Numerical, dur_gradhessian'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.gradhessian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_gradhessian'_Symbolic, dur_gradhessian'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.gradhessian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: laplacian'"
    let res_laplacian'_AD_Forward, dur_laplacian'_AD_Forward = 0., 0.
    let res_laplacian'_AD_Forward2, dur_laplacian'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian'_AD_ForwardG, dur_laplacian'_AD_ForwardG = 0., 0.
    let res_laplacian'_AD_ForwardGH, dur_laplacian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian'_AD_ForwardN, dur_laplacian'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian'_AD_Reverse, dur_laplacian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian'_Numerical, dur_laplacian'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.laplacian' (fun x -> (x.[0] * (sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]) xv)
    let res_laplacian'_Symbolic, dur_laplacian'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.laplacian' <@ fun x0 x1 x2 -> (x0 * (sqrt (x1 + x2)) * (log x2)) ** x1 @> xv)

    printfn "Running benchmark: jacobian'"
    let res_jacobian'_AD_Forward, dur_jacobian'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_AD_Forward2, dur_jacobian'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_AD_ForwardG, dur_jacobian'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_AD_ForwardGH, dur_jacobian'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_AD_ForwardN, dur_jacobian'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_AD_Reverse, dur_jacobian'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_Numerical, dur_jacobian'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobian' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobian'_Symbolic, dur_jacobian'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.jacobian' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @> xv)

    printfn "Running benchmark: jacobianv'"
    let res_jacobianv'_AD_Forward, dur_jacobianv'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv'_AD_Forward2, dur_jacobianv'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv'_AD_ForwardG, dur_jacobianv'_AD_ForwardG = 0., 0.
    let res_jacobianv'_AD_ForwardGH, dur_jacobianv'_AD_ForwardGH = 0., 0.
    let res_jacobianv'_AD_ForwardN, dur_jacobianv'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv'_AD_Reverse, dur_jacobianv'_AD_Reverse = 0., 0.
    let res_jacobianv'_Numerical, dur_jacobianv'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobianv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv v)
    let res_jacobianv'_Symbolic, dur_jacobianv'_Symbolic = 0., 0.

    printfn "Running benchmark: jacobianT'"
    let res_jacobianT'_AD_Forward, dur_jacobianT'_AD_Forward = duration n (fun () -> DiffSharp.AD.Forward.ForwardOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_AD_Forward2, dur_jacobianT'_AD_Forward2 = duration n (fun () -> DiffSharp.AD.Forward2.Forward2Ops.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_AD_ForwardG, dur_jacobianT'_AD_ForwardG = duration n (fun () -> DiffSharp.AD.ForwardG.ForwardGOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_AD_ForwardGH, dur_jacobianT'_AD_ForwardGH = duration n (fun () -> DiffSharp.AD.ForwardGH.ForwardGHOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_AD_ForwardN, dur_jacobianT'_AD_ForwardN = duration n (fun () -> DiffSharp.AD.ForwardN.ForwardNOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_AD_Reverse, dur_jacobianT'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_Numerical, dur_jacobianT'_Numerical = duration n (fun () -> DiffSharp.Numerical.NumericalOps.jacobianT' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv)
    let res_jacobianT'_Symbolic, dur_jacobianT'_Symbolic = duration nsymbolic (fun () -> DiffSharp.Symbolic.SymbolicOps.jacobianT' <@ fun x0 x1 x2 -> [|(sin x0) ** x1; sqrt (x1 + x2); log (x0 * x2)|] @> xv)

    printfn "Running benchmark: jacobianTv'"
    let res_jacobianTv'_AD_Forward, dur_jacobianTv'_AD_Forward = 0., 0.
    let res_jacobianTv'_AD_Forward2, dur_jacobianTv'_AD_Forward2 = 0., 0.
    let res_jacobianTv'_AD_ForwardG, dur_jacobianTv'_AD_ForwardG = 0., 0.
    let res_jacobianTv'_AD_ForwardGH, dur_jacobianTv'_AD_ForwardGH = 0., 0.
    let res_jacobianTv'_AD_ForwardN, dur_jacobianTv'_AD_ForwardN = 0., 0.
    let res_jacobianTv'_AD_Reverse, dur_jacobianTv'_AD_Reverse = duration n (fun () -> DiffSharp.AD.Reverse.ReverseOps.jacobianTv' (fun x -> [|(sin x.[0]) ** x.[1]; sqrt (x.[1] + x.[2]); log (x.[0] * x.[2])|]) xv u)
    let res_jacobianTv'_Numerical, dur_jacobianTv'_Numerical = 0., 0.
    let res_jacobianTv'_Symbolic, dur_jacobianTv'_Symbolic = 0., 0.

    //
    //
    //
    //
    //
    //

    let finished = System.DateTime.Now
    printfn "Benchmarking finished: %A\n" finished
    printfn "Total duration: %A\n" (finished - started)

    printfn "Writing results to file: %s" file

    let row_originals = Vector.Create([| dur_fss; dur_fss; dur_fss; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvv; dur_fvv; dur_fvv; dur_fvv |])
    let row_AD_Forward = Vector.Create([| dur_diff_AD_Forward; dur_diff2_AD_Forward; dur_diffn_AD_Forward; dur_grad_AD_Forward; dur_gradv_AD_Forward; dur_hessian_AD_Forward; dur_gradhessian_AD_Forward; dur_laplacian_AD_Forward; dur_jacobian_AD_Forward; dur_jacobianv_AD_Forward; dur_jacobianT_AD_Forward; dur_jacobianTv_AD_Forward |]) / row_originals
    let row_AD_Forward2 = Vector.Create([| dur_diff_AD_Forward2; dur_diff2_AD_Forward2; dur_diffn_AD_Forward2; dur_grad_AD_Forward2; dur_gradv_AD_Forward2; dur_hessian_AD_Forward2; dur_gradhessian_AD_Forward2; dur_laplacian_AD_Forward2; dur_jacobian_AD_Forward2; dur_jacobianv_AD_Forward2; dur_jacobianT_AD_Forward2; dur_jacobianTv_AD_Forward2 |]) / row_originals
    let row_AD_ForwardG = Vector.Create([| dur_diff_AD_ForwardG; dur_diff2_AD_ForwardG; dur_diffn_AD_ForwardG; dur_grad_AD_ForwardG; dur_gradv_AD_ForwardG; dur_hessian_AD_ForwardG; dur_gradhessian_AD_ForwardG; dur_laplacian_AD_ForwardG; dur_jacobian_AD_ForwardG; dur_jacobianv_AD_ForwardG; dur_jacobianT_AD_ForwardG; dur_jacobianTv_AD_ForwardG |]) / row_originals
    let row_AD_ForwardGH = Vector.Create([| dur_diff_AD_ForwardGH; dur_diff2_AD_ForwardGH; dur_diffn_AD_ForwardGH; dur_grad_AD_ForwardGH; dur_gradv_AD_ForwardGH; dur_hessian_AD_ForwardGH; dur_gradhessian_AD_ForwardGH; dur_laplacian_AD_ForwardGH; dur_jacobian_AD_ForwardGH; dur_jacobianv_AD_ForwardGH; dur_jacobianT_AD_ForwardGH; dur_jacobianTv_AD_ForwardGH |]) / row_originals
    let row_AD_ForwardN = Vector.Create([| dur_diff_AD_ForwardN; dur_diff2_AD_ForwardN; dur_diffn_AD_ForwardN; dur_grad_AD_ForwardN; dur_gradv_AD_ForwardN; dur_hessian_AD_ForwardN; dur_gradhessian_AD_ForwardN; dur_laplacian_AD_ForwardN; dur_jacobian_AD_ForwardN; dur_jacobianv_AD_ForwardN; dur_jacobianT_AD_ForwardN; dur_jacobianTv_AD_ForwardN |]) / row_originals
    let row_AD_Reverse = Vector.Create([| dur_diff_AD_Reverse; dur_diff2_AD_Reverse; dur_diffn_AD_Reverse; dur_grad_AD_Reverse; dur_gradv_AD_Reverse; dur_hessian_AD_Reverse; dur_gradhessian_AD_Reverse; dur_laplacian_AD_Reverse; dur_jacobian_AD_Reverse; dur_jacobianv_AD_Reverse; dur_jacobianT_AD_Reverse; dur_jacobianTv_AD_Reverse |]) / row_originals
    let row_Numerical = Vector.Create([| dur_diff_Numerical; dur_diff2_Numerical; dur_diffn_Numerical; dur_grad_Numerical; dur_gradv_Numerical; dur_hessian_Numerical; dur_gradhessian_Numerical; dur_laplacian_Numerical; dur_jacobian_Numerical; dur_jacobianv_Numerical; dur_jacobianT_Numerical; dur_jacobianTv_Numerical |]) / row_originals
    let row_Symbolic = Vector.Create([| dur_diff_Symbolic; dur_diff2_Symbolic; dur_diffn_Symbolic; dur_grad_Symbolic; dur_gradv_Symbolic; dur_hessian_Symbolic; dur_gradhessian_Symbolic; dur_laplacian_Symbolic; dur_jacobian_Symbolic; dur_jacobianv_Symbolic; dur_jacobianT_Symbolic; dur_jacobianTv_Symbolic |]) / row_originals

    //let benchmark = Matrix.Create([| row_AD_Forward; row_AD_Forward2; row_AD_ForwardG; row_AD_ForwardGH; row_AD_ForwardN; row_AD_Reverse; row_Numerical |])
    let benchmark = Matrix.Create([| row_AD_Forward; row_AD_Forward2; row_AD_ForwardG; row_AD_ForwardGH; row_AD_ForwardN; row_AD_Reverse; row_Numerical; row_Symbolic |])

    let row_originals' = Vector.Create([| dur_fss; dur_fss; dur_fss; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvs; dur_fvv; dur_fvv; dur_fvv; dur_fvv |])
    let row_AD_Forward' = Vector.Create([| dur_diff'_AD_Forward; dur_diff2'_AD_Forward; dur_diffn'_AD_Forward; dur_grad'_AD_Forward; dur_gradv'_AD_Forward; dur_hessian'_AD_Forward; dur_gradhessian'_AD_Forward; dur_laplacian'_AD_Forward; dur_jacobian'_AD_Forward; dur_jacobianv'_AD_Forward; dur_jacobianT'_AD_Forward; dur_jacobianTv'_AD_Forward |]) / row_originals'
    let row_AD_Forward2' = Vector.Create([| dur_diff'_AD_Forward2; dur_diff2'_AD_Forward2; dur_diffn'_AD_Forward2; dur_grad'_AD_Forward2; dur_gradv'_AD_Forward2; dur_hessian'_AD_Forward2; dur_gradhessian'_AD_Forward2; dur_laplacian'_AD_Forward2; dur_jacobian'_AD_Forward2; dur_jacobianv'_AD_Forward2; dur_jacobianT'_AD_Forward2; dur_jacobianTv'_AD_Forward2 |]) / row_originals'
    let row_AD_ForwardG' = Vector.Create([| dur_diff'_AD_ForwardG; dur_diff2'_AD_ForwardG; dur_diffn'_AD_ForwardG; dur_grad'_AD_ForwardG; dur_gradv'_AD_ForwardG; dur_hessian'_AD_ForwardG; dur_gradhessian'_AD_ForwardG; dur_laplacian'_AD_ForwardG; dur_jacobian'_AD_ForwardG; dur_jacobianv'_AD_ForwardG; dur_jacobianT'_AD_ForwardG; dur_jacobianTv'_AD_ForwardG |]) / row_originals'
    let row_AD_ForwardGH' = Vector.Create([| dur_diff'_AD_ForwardGH; dur_diff2'_AD_ForwardGH; dur_diffn'_AD_ForwardGH; dur_grad'_AD_ForwardGH; dur_gradv'_AD_ForwardGH; dur_hessian'_AD_ForwardGH; dur_gradhessian'_AD_ForwardGH; dur_laplacian'_AD_ForwardGH; dur_jacobian'_AD_ForwardGH; dur_jacobianv'_AD_ForwardGH; dur_jacobianT'_AD_ForwardGH; dur_jacobianTv'_AD_ForwardGH |]) / row_originals'
    let row_AD_ForwardN' = Vector.Create([| dur_diff'_AD_ForwardN; dur_diff2'_AD_ForwardN; dur_diffn'_AD_ForwardN; dur_grad'_AD_ForwardN; dur_gradv'_AD_ForwardN; dur_hessian'_AD_ForwardN; dur_gradhessian'_AD_ForwardN; dur_laplacian'_AD_ForwardN; dur_jacobian'_AD_ForwardN; dur_jacobianv'_AD_ForwardN; dur_jacobianT'_AD_ForwardN; dur_jacobianTv'_AD_ForwardN |]) / row_originals'
    let row_AD_Reverse' = Vector.Create([| dur_diff'_AD_Reverse; dur_diff2'_AD_Reverse; dur_diffn'_AD_Reverse; dur_grad'_AD_Reverse; dur_gradv'_AD_Reverse; dur_hessian'_AD_Reverse; dur_gradhessian'_AD_Reverse; dur_laplacian'_AD_Reverse; dur_jacobian'_AD_Reverse; dur_jacobianv'_AD_Reverse; dur_jacobianT'_AD_Reverse; dur_jacobianTv'_AD_Reverse |]) / row_originals'
    let row_Numerical' = Vector.Create([| dur_diff'_Numerical; dur_diff2'_Numerical; dur_diffn'_Numerical; dur_grad'_Numerical; dur_gradv'_Numerical; dur_hessian'_Numerical; dur_gradhessian'_Numerical; dur_laplacian'_Numerical; dur_jacobian'_Numerical; dur_jacobianv'_Numerical; dur_jacobianT'_Numerical; dur_jacobianTv'_Numerical |]) / row_originals'
    let row_Symbolic' = Vector.Create([| dur_diff'_Symbolic; dur_diff2'_Symbolic; dur_diffn'_Symbolic; dur_grad'_Symbolic; dur_gradv'_Symbolic; dur_hessian'_Symbolic; dur_gradhessian'_Symbolic; dur_laplacian'_Symbolic; dur_jacobian'_Symbolic; dur_jacobianv'_Symbolic; dur_jacobianT'_Symbolic; dur_jacobianTv'_Symbolic |]) / row_originals'

    //let benchmark' = Matrix.Create([| row_AD_Forward'; row_AD_Forward2'; row_AD_ForwardG'; row_AD_ForwardGH'; row_AD_ForwardN'; row_AD_Reverse'; row_Numerical' |])
    let benchmark' = Matrix.Create([| row_AD_Forward'; row_AD_Forward2'; row_AD_ForwardG'; row_AD_ForwardGH'; row_AD_ForwardN'; row_AD_Reverse'; row_Numerical'; row_Symbolic' |])

    let stream = new System.IO.StreamWriter(file, false)
    stream.WriteLine("DiffSharp\r\n")
    stream.WriteLine(sprintf "Benchmarking module version: %s" benchmarkver)
    stream.WriteLine(sprintf "DiffSharp library version: %s\r\n" diffsharpver)
    stream.WriteLine(sprintf "OS: %s" os)
    stream.WriteLine(sprintf ".NET CLR version: %s" clr)
    stream.WriteLine(sprintf "CPU: %s" cpu)
    stream.WriteLine(sprintf "RAM: %s\r\n" ram)
    stream.WriteLine(sprintf "Repetitions: %A\r\n" n)
    stream.WriteLine(sprintf "Benchmarking started: %A" started)
    stream.WriteLine(sprintf "Benchmarking finished: %A" finished)
    stream.WriteLine(sprintf "Total duration: %A\r\n" (finished - started))
    
    stream.WriteLine("Benchmark A\r\n")
    stream.WriteLine("Columns: {diff, diff2, diffn, grad, gradv, hessian, gradhessian, laplacian, jacobian, jacobianv, jacobianT, jacobianTv}")
    stream.WriteLine("Rows: {DiffSharp.AD.Forward, DiffSharp.AD.Forward2, DiffSharp.AD.ForwardG, DiffSharp.AD.ForwardGH, DiffSharp.AD.ForwardN, DiffSharp.AD.Reverse, DiffSharp.Numerical, DiffSharp.Symbolic}\r\n")
    stream.WriteLine(benchmark.ToMathematicaString())

    stream.WriteLine("\r\nBenchmark B\r\n")
    stream.WriteLine("Columns: {diff', diff2', diffn', grad', gradv', hessian', gradhessian', laplacian', jacobian', jacobianv', jacobianT', jacobianTv'}")
    stream.WriteLine("Rows: {DiffSharp.AD.Forward, DiffSharp.AD.Forward2, DiffSharp.AD.ForwardG, DiffSharp.AD.ForwardGH, DiffSharp.AD.ForwardN, DiffSharp.AD.Reverse, DiffSharp.Numerical, DiffSharp.Symbolic}\r\n")
    stream.WriteLine(benchmark'.ToMathematicaString())
    stream.Flush()
    stream.Close()

    0 // return an integer exit code
