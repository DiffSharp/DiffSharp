namespace DiffSharp.RawTensor
open DiffSharp.Util

type RawTensorFloat32CPUBase(value: float32[], shape:int[]) =
    inherit RawTensor(value, shape, Float32, CPU, CPUBase)

    member private t.GetItem (index:int[]) =
        // if index.Length <> t.Dim then invalidArg "index" (sprintf "Expecting a %i-dimensional index" t.Dim)
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) t.Shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        let tvalue = t.Value:?>float32[]
        tvalue.[flatIndex]

    override t.Create(value) = upcast RawTensorFloat32CPUBase(toFlatArrayAndShape value)
    override t.CreateWithShape(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPUBase([|value|], [||])
        | _ -> upcast RawTensorFloat32CPUBase(Array.create (shape |> Array.reduce (*)) value, shape)
    override t.Zero() = upcast RawTensorFloat32CPUBase([|0.f|], [||])
    override t.Zeros(shape) = t.CreateWithShape(0.f, shape)
    override t.One() = upcast RawTensorFloat32CPUBase([|1.f|], [||])
    override t.Ones(shape) = t.CreateWithShape(1.f, shape)
    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        match t.Dim with
        | 0 -> sprintf "%A" (t.Value:?>float32[]).[0]
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(sprintf "%A" (t.GetItem(globalCoords))) |> ignore
                        prefix <- "; "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- "; "
                    sb.Append("]") |> ignore
            print t.Shape [||]
            sb.ToString()

    override t.ToValue() =
        match t.Dim with
        | 0 -> upcast (t.Value:?>float32[]).[0]
        | _ -> invalidOp (sprintf "Cannot convert %Ad Tensor to single value" t.Dim)

    override t.ToArray() =
        match t.Dim with
        | 0 -> invalidOp "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.GetItem([|i|]))
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.GetItem([|i; j|]))
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.GetItem([|i; j; k|]))
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.GetItem([|i; j; k; l|]))
        | _ -> invalidOp (sprintf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape)

    override t1.Equals(t2:RawTensor) = 
        match t2 with
        | :? RawTensorFloat32CPUBase as t2 -> arraysEqual (t1.Value:?>float32[]) (t2.Value:?>float32[])
        | _ -> failwith (sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2)

    override t1.AddTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (+) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.AddTS(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map ((+) t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t.Sum() =
        let tvalue = t.Value:?>float32[]
        let result = Array.reduce (+) tvalue
        upcast RawTensorFloat32CPUBase([|result|], [||])