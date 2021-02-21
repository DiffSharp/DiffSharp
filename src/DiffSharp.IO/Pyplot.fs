namespace DiffSharp.IO

open DiffSharp
open System.IO
open System.Diagnostics


[<AutoOpen>]
module helpers =
    let tensorToPython (t:Tensor) =
        let printVal (x:scalar) = 
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
        let sb = System.Text.StringBuilder()
        match t.dim with
        | 0 -> 
            sb.Append(printVal (t.toScalar())) |> ignore
        | _ ->
            let rec print (shape:Shape) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(printVal (t.Item(globalCoords))) |> ignore
                        prefix <- ", "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- ", "
                    sb.Append("]") |> ignore
            print t.shape [||]
        sb.ToString()

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
            failwithf "Error or timeout while running process %s" executable


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
        add(sprintf "plt.plot(%s, %s, alpha=%A, label='%s')" (tensorToPython x) (tensorToPython y) alpha label)
    member p.plot(y:Tensor, ?alpha, ?label) =
        let x = dsharp.arangeLike(y, y.nelement)
        p.plot(x, y, ?alpha=alpha, ?label=label)
    member p.hist(x:Tensor, ?weights, ?bins, ?density, ?label) =
        let weights = defaultArg weights (dsharp.onesLike(x))
        let bins = defaultArg bins 10
        let density = defaultArg density false
        let label = defaultArg label ""
        add(sprintf "plt.hist(%s, weights=%s, bins=%A, density=%s, label='%s')" (tensorToPython x) (tensorToPython weights) bins (if density then "True" else "False") label)
    member _.figure(?figSize) = 
        let figSize = defaultArg figSize (6.4, 4.8)
        add(sprintf "plt.figure(figsize=(%A,%A))" (fst figSize) (snd figSize))
    member _.legend() = add("plt.legend()")
    member _.tightLayout() = add("plt.tight_layout()")
    member _.savefig(fileName) =
        add(sprintf "plt.savefig('%s')" fileName)
        runScript pythonExecutable lines timeoutMilliseconds
        reset()