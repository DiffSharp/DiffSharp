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
    
    static member Create(value:obj):RawTensor = 
        let array, shape = value |> flatArrayAndShape<float32>
        if notNull array then 
            upcast RawTensorFloat32CPUBase(array, shape)
        else 
            let array, shape = value |> flatArrayAndShape<double>
            if notNull array then 
                upcast RawTensorFloat32CPUBase(array |> Array.map float32, shape)
            else
                let array, shape = value |> flatArrayAndShape<int>
                if notNull array then 
                    upcast RawTensorFloat32CPUBase(array |> Array.map float32, shape)
                else
                    invalidArg "value" "Cannot convert value to RawTensorFloat32CPUBase"
    static member Zeros(shape:int[]):RawTensor = upcast RawTensorFloat32CPUBase(Array.create (shapeLength shape) 0.f, shape)
    static member Ones(shape:int[]):RawTensor = upcast RawTensorFloat32CPUBase(Array.create (shapeLength shape) 1.f, shape)
    static member Random(shape:int[]):RawTensor = upcast RawTensorFloat32CPUBase(Array.init (shapeLength shape) (fun _ -> float32 (Random.Uniform())), shape)
    static member RandomNormal(shape:int[]):RawTensor = upcast RawTensorFloat32CPUBase(Array.init (shapeLength shape) (fun _ -> float32 (Random.Normal())), shape)

    override t.Create(value) = RawTensorFloat32CPUBase.Create(value)
    override t.CreateWithShape(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPUBase([|value|], [||])
        | _ -> upcast RawTensorFloat32CPUBase(Array.create (shape |> Array.reduce (*)) value, shape)
    override t.Zero() = upcast RawTensorFloat32CPUBase([|0.f|], [||])
    override t.Zeros(shape) = RawTensorFloat32CPUBase.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPUBase([|1.f|], [||])
    override t.Ones(shape) = RawTensorFloat32CPUBase.Ones( shape)
    override t.Random(shape) = RawTensorFloat32CPUBase.Random(shape)
    override t.RandomNormal(shape) = RawTensorFloat32CPUBase.RandomNormal(shape)

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
        | :? RawTensorFloat32CPUBase as t2 -> (t1.Value:?>float32[]) = (t2.Value:?>float32[])
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) =
        let tolerance = float32 <| tolerance
        match t2 with
        | :? RawTensorFloat32CPUBase as t2 -> arraysApproximatelyEqual tolerance (t1.Value:?>float32[]) (t2.Value:?>float32[])
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t.Stack(tensors) =
        let tensors = tensors |> Seq.toList
        let values, shapes = tensors |> List.map (fun t -> t.Value:?>float32[], t.Shape) |> List.unzip
        if not (allEqual shapes) then invalidArg "tensors" "Expecting Tensors with same shape"
        let n = tensors |> List.length
        let m = shapeLength shapes.[0]
        let result = Array.create (n * m) 0.f
        for i=0 to n-1 do
            for j=0 to m-1 do
                result.[i*m+j] <-values.[i].[j]
        upcast RawTensorFloat32CPUBase(result, Array.append [|n|] shapes.[0])

    override t1.AddTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (+) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.AddTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map ((+) t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.AddT2T1(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.SubTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (-) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.SubT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map ((-) t1value) t2value
        upcast RawTensorFloat32CPUBase(result, t2.Shape)

    override t1.SubTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t - t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.MulTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (*) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.MulTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map ((*) t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.DivTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (/) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.DivT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map ((/) t1value) t2value
        upcast RawTensorFloat32CPUBase(result, t2.Shape)

    override t1.DivTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t / t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.PowTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 ( ** ) t1value t2value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.PowT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map (fun t -> t1value ** t) t2value
        upcast RawTensorFloat32CPUBase(result, t2.Shape)

    override t1.PowTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t1.MatMulT2T2(t2) =
        if t1.Dim <> 2 || t2.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        if t1cols <> t2rows then invalidOp <| sprintf "Cannot multiply Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]        
        let result = Array2D.init t1rows t2cols (fun i j -> Array.sumBy (fun k -> t1value.[i*t1cols + k] * t2value.[k*t2cols + j]) [|0..(t2rows-1)|] )
        RawTensorFloat32CPUBase.Create(result)
        
    override t.NegT() =
        let tvalue = t.Value:?>float32[]
        let result = Array.map (~-) tvalue
        upcast RawTensorFloat32CPUBase(result, t.Shape)

    override t.SumT() =
        let tvalue = t.Value:?>float32[]
        let result = Array.reduce (+) tvalue
        upcast RawTensorFloat32CPUBase([|result|], [||])
    
    override t.SumT2Dim0() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>float32[]
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> tvalue.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        upcast RawTensorFloat32CPUBase(result, resultShape)

    override t.TransposeT2() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>float32[]
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> tvalue.[j*tcols + i])
        RawTensorFloat32CPUBase.Create(result)

    override t.SignT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map (sign >> float32)
        upcast RawTensorFloat32CPUBase(result, t.Shape)

    override t.AbsT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map abs
        upcast RawTensorFloat32CPUBase(result, t.Shape)

    override t1.ReLUT() =
        let t1value = t1.Value:?>float32[]
        let result = Array.map (max 0.f) t1value
        upcast RawTensorFloat32CPUBase(result, t1.Shape)

    override t.ExpT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map exp
        upcast RawTensorFloat32CPUBase(result, t.Shape)

    override t.LogT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map log
        upcast RawTensorFloat32CPUBase(result, t.Shape)                