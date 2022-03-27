// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Util

open DiffSharp
open System.IO
open System.Diagnostics


[<AutoOpen>]
module helpers =
    let printVal (x:scalar) = 
        let s = 
            match x.GetTypeCode() with
            | System.TypeCode.Single -> sprintf "%f" (x.toSingle())
            | System.TypeCode.Double -> sprintf "%f" (x.toDouble())
            | System.TypeCode.Int32 -> sprintf "%d" (x.toInt32())
            | System.TypeCode.Int64 -> sprintf "%d" (x.toInt64())
            | System.TypeCode.Byte -> sprintf "%d" (x.toByte())
            | System.TypeCode.SByte -> sprintf "%d" (x.toSByte())
            | System.TypeCode.Int16 -> sprintf "%d" (x.toInt16())
            | System.TypeCode.Boolean -> if (x.toBool()) then "True" else "False"
            | _ -> x.ToString()
        s.Replace("NaN", "float('nan')").Replace("Infinity", "float('inf')")

    let toPython (v:obj) =
        match v with
        | :? bool as b -> printVal b
        | :? Tensor as t ->
            let sb = System.Text.StringBuilder()
            match t.dim with
            | 0 -> 
                sb.Append(printVal (t.toScalar())) |> ignore
            | _ ->
                let rec print (shape:Shape) externalCoords = 
                    if shape.Length = 1 then
                        sb.Append("[") |> ignore
                        let mutable prefix = ""
                        for i=0 to shape[0]-1 do
                            let globalCoords = Array.append externalCoords [|i|]
                            sb.Append(prefix) |> ignore
                            sb.Append(printVal (t.Item(globalCoords))) |> ignore
                            prefix <- ", "
                        sb.Append("]") |> ignore
                    else
                        sb.Append("[") |> ignore
                        let mutable prefix = ""
                        for i=0 to shape[0]-1 do
                            sb.Append(prefix) |> ignore
                            print shape[1..] (Array.append externalCoords [|i|])
                            prefix <- ", "
                        sb.Append("]") |> ignore
                print t.shape [||]
            sb.ToString()
        | _ -> v.ToString()

    let runScript executable lines timeoutMilliseconds =
        let fileName = Path.GetTempFileName()
        File.WriteAllLines(fileName, lines)
        let success =
            try
                let p = Process.Start(executable, fileName)
                p.WaitForExit(timeoutMilliseconds)
            with
                | _ -> false
        if not success then
            printfn "Warning: cannot plot due to error or timeout while running %s" executable


// This is a lightweight wrapper roughly compatible with the matplotlib.pyplot API
// https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot
// - It works by creating little Python scripts on the fly for plotting, then running the Python
// interpreter to save the resulting plots to a file (e.g., png, pdf)
// - The intention is to cover the typical use cases for a machine learning user (e.g., line plots, histograms)
// - We expect to have a local Python distribution in the machine, with matplotlib package installed
// - This design is cross-platform and easier to maintain longer-term compared with deeper Python integration
// solutions like pythonnet http://pythonnet.github.io/ which need platform-specific and Python-version-specific 
// dependency packages that seem to be not well maintained
//
// Example:
//
// let y1 = dsharp.randn(10)
// let y2 = dsharp.randn(10)
// let plt = Pyplot()
// plt.figure((10., 6.))
// plt.plot(y1, label="first")
// plt.plot(y2, label="second", alpha=0.5)
// plt.legend()
// plt.tightLayout()
// plt.savefig("test.png")

type Pyplot(?pythonExecutable, ?timeoutMilliseconds) =
    let pythonExecutable = defaultArg pythonExecutable "python"
    let timeoutMilliseconds = defaultArg timeoutMilliseconds 10000
    let preamble = "import matplotlib.pyplot as plt"
    let mutable lines = [|preamble|]
    let reset () = lines <- [|preamble|]
    let add l = lines <- Array.append lines [|l|]

    member _.script = lines |> Array.fold (fun r s -> r + s + "\n") ""
    member _.addPython(line) = add(line)
    member _.plot(x:Tensor, y:Tensor, ?alpha, ?label) =
        if x.dim <> 1 || y.dim <> 1 then failwithf "Expecting tensors x (%A) and y (%A) to be 1d" x.shape y.shape
        let alpha = defaultArg alpha 1.
        let label = defaultArg label ""
        add(sprintf "plt.plot(%s, %s, alpha=%A, label='%s')" (toPython x) (toPython y) alpha label)
    member p.plot(y:Tensor, ?alpha, ?label) =
        if y.dim <> 1 then failwithf "Expecting tensor y (%A) to be 1d" y.shape
        let x = dsharp.arangeLike(y, y.nelement)
        p.plot(x, y, ?alpha=alpha, ?label=label)
    member p.hist(x:Tensor, ?weights, ?bins, ?density, ?label) =
        if x.dim <> 1 then failwithf "Expecting tensor x (%A) to be 1d" x.shape
        let weights = defaultArg weights (dsharp.onesLike(x))
        if weights.dim <> 1 then failwithf "Expecting tensor weights (%A) to be 1d" weights.shape
        let bins = defaultArg bins 10
        let density = defaultArg density false
        let label = defaultArg label ""
        add(sprintf "plt.hist(%s, weights=%s, bins=%A, density=%s, label='%s')" (toPython x) (toPython weights) bins (toPython density) label)
    member _.figure(?figSize) = 
        let figSize = defaultArg figSize (6.4, 4.8)
        add(sprintf "plt.figure(figsize=(%A,%A))" (fst figSize) (snd figSize))
    member _.legend() = add("plt.legend()")
    member _.tightLayout() = add("plt.tight_layout()")
    member _.xlabel(label) = add(sprintf "plt.xlabel('%s')" label)
    member _.ylabel(label) = add(sprintf "plt.ylabel('%s')" label)
    member _.xscale(value) = add(sprintf "plt.xscale('%s')" value)
    member _.yscale(value) = add(sprintf "plt.yscale('%s')" value)
    member _.savefig(fileName) =
        add(sprintf "plt.savefig('%s')" fileName)
        runScript pythonExecutable lines timeoutMilliseconds
        reset()