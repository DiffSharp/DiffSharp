namespace DiffSharp.Backend.Reference
open DiffSharp
open DiffSharp.Backend
open DiffSharp.Util
open System

type RawTensorFloat32CPU(value: float32[], shape:int[]) =
    inherit RawTensor(value, shape, Float32, CPU, Backend.Reference)

    member private t.IndexToFlatIndex(index:int[]) =
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) t.Shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        flatIndex
    
    member private t.FlatIndexToIndex(flatIndex:int) =
        let index = Array.create t.Dim 0
        let mutable mul = t.Nelement
        let mutable fi = flatIndex
        for i=t.Dim downto 1 do
            mul <- mul / t.Shape.[t.Dim-i]
            index.[i-1] <- fi / mul
            fi <- fi - index.[i-1] * mul
        index |> Array.rev

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            if index.Length <> t.Dim then invalidArg "index" (sprintf "Expecting a %id index" t.Dim)
            let tvalue = t.Value:?>float32[]
            tvalue.[t.IndexToFlatIndex(index)]

    override t.GetItem(index:int[]) = RawTensorFloat32CPU.Create(t.[index])
    override t.GetSlice(bounds:int[,]) =
        // if bounds.GetLength(0) <> t.Dim then invalidArg "bounds" (sprintf "Expecting %i-by-2 bounds" t.Dim)
        // printfn "%A" bounds
        let shape = Array.init (bounds.GetLength(0)) (fun i -> bounds.[i,1] - bounds.[i,0] + 1) |> shapeSqueeze -1
        // printfn "%A" shape
        let array = Array.create (shapeLength shape) 0.f
        let mutable arrayi = 0
        let rec slice (bounds:int[,]) externalCoords =
            if bounds.GetLength(0) = 1 then
                for i=bounds.[0,0] to bounds.[0,1] do
                    // printfn "inner %A" i
                    let globalCoords = Array.append externalCoords [|i|]
                    array.[arrayi] <- t.[globalCoords]
                    arrayi <- arrayi + 1
            else
                for i=bounds.[0,0] to bounds.[0,1] do
                    // printfn "outer %A" i
                    slice bounds.[1..,*] (Array.append externalCoords [|i|])
        slice bounds [||]
        upcast RawTensorFloat32CPU(array, shape)

    override t1.CompareTo(t2) =
        compare (t1.ToValue():?>float32) (t2.ToValue():?>float32)
    override t.Create(value) = RawTensorFloat32CPU.Create(value)
    override t.Create(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPU([|value|], [||])
        | _ -> upcast RawTensorFloat32CPU(Array.create (shapeLength shape) value, shape)
    override t.Zero() = upcast RawTensorFloat32CPU([|0.f|], [||])
    override t.Zeros(shape) = RawTensorFloat32CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPU([|1.f|], [||])
    override t.Ones(shape) = RawTensorFloat32CPU.Ones(shape)
    override t.Random(shape) = RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = RawTensorFloat32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorFloat32CPU.RandomMultinomial(t, numSamples)
    static member RandomMultinomial(probs:RawTensor, numSamples:int):RawTensor =
        if probs.Dim < 1 || probs.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" probs.Shape
        if probs.Dim = 1 then
            let p = probs.Value :?> float32[] |> Array.map float
            let result = [|for i=0 to numSamples-1 do yield float32 (Random.ChoiceIndex(p))|]
            upcast RawTensorFloat32CPU(result, [|numSamples|])
        else
            let p = probs.ToArray() :?> float32[,] |> Array2D.map float
            let result = Array2D.init (p.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(p.[i,*]))
            RawTensorFloat32CPU.Create(result)

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
                        sb.Append(sprintf "%A" (t.[globalCoords])) |> ignore
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
        | _ -> invalidOp (sprintf "Cannot convert %Ad Tensor to scalar" t.Dim)

    override t.ToArray() =
        match t.Dim with
        | 0 -> invalidOp "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> invalidOp (sprintf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape)

    override t1.Equals(t2:RawTensor) = 
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && (t1.Value:?>float32[]) = (t2.Value:?>float32[])
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) =
        let tolerance = float32 <| tolerance
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && arraysApproximatelyEqual tolerance (t1.Value:?>float32[]) (t2.Value:?>float32[])
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override __.StackTs(tensors) =
        let tensors = tensors |> Seq.toList
        let values, shapes = tensors |> List.map (fun t -> t.Value:?>float32[], t.Shape) |> List.unzip
        if not (allEqual shapes) then invalidArg "tensors" "Expecting Tensors with same shape"
        let n = tensors |> List.length
        let m = shapeLength shapes.[0]
        let result = Array.create (n * m) 0.f
        for i=0 to n-1 do
            for j=0 to m-1 do
                result.[i*m+j] <-values.[i].[j]
        upcast RawTensorFloat32CPU(result, Array.append [|n|] shapes.[0])

    override t.UnstackT() =
        if t.Dim < 1 then invalidOp "Cannot unstack scalar Tensor (dim < 1)"
        let tvalue = t.Value:?>float32[]
        let n = t.Shape.[0]
        let unstackedShape = if t.Dim = 1 then [||] else t.Shape |> Array.skip 1
        let unstackedLength = shapeLength unstackedShape
        Seq.init n (fun i -> Array.init unstackedLength (fun j -> tvalue.[i*unstackedLength+j]))
        |> Seq.map (fun v -> upcast RawTensorFloat32CPU(v, unstackedShape))

    override t1.LtTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (fun t1 t2 -> if t1 < t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GtTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (fun t1 t2 -> if t1 > t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.LeTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (fun t1 t2 -> if t1 <= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GeTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (fun t1 t2 -> if t1 >= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.MaxIndexT() =
        let tvalue = t.Value:?>float32[]
        t.FlatIndexToIndex(maxIndex tvalue)

    override t.MinIndexT() =
        let tvalue = t.Value:?>float32[]
        t.FlatIndexToIndex(minIndex tvalue)

    override t1.AddTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (+) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map ((+) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddT2T1(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTTSlice(location:int[], t2) =
        // if not (shapeContains t1.Shape t2.Shape) then failwithf "Expecting t1.Shape to contain t2.Shape, received %A, %A" t1.Shape t2.Shape
        let t1value = t1.Value:?>float32[]
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = Array.copy t1value
        let shape2 = shapeUnsqueezeAs t2.Shape t1.Shape
        let rec add (shape2:int[]) externalCoords =
            if shape2.Length = 1 then
                for i=0 to shape2.[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let t1Coords = Array.map2 (+) globalCoords location
                    let t1FlatIndex = t1.IndexToFlatIndex(t1Coords)
                    result.[t1FlatIndex] <- result.[t1FlatIndex] + t2.[globalCoords]
            else
                for i=0 to shape2.[0]-1 do
                    add (shape2.[1..]) (Array.append externalCoords [|i|])
        add shape2 [||]
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (-) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map ((-) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.SubTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t - t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (*) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map ((*) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 (/) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map ((/) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.DivTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t / t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowTT(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]
        let result = Array.map2 ( ** ) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowT0T(t2) =
        let t1value = (t1.Value:?>float32[]).[0]
        let t2value = (t2.Value:?>float32[])
        let result = Array.map (fun t -> t1value ** t) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.PowTT0(t2) =
        let t1value = t1.Value:?>float32[]
        let t2value = (t2.Value:?>float32[]).[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MatMulT2T2(t2) =
        if t1.Dim <> 2 || t2.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        if t1cols <> t2rows then invalidOp <| sprintf "Cannot multiply Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1value = t1.Value:?>float32[]
        let t2value = t2.Value:?>float32[]        
        let result = Array2D.init t1rows t2cols (fun i j -> Array.sumBy (fun k -> t1value.[i*t1cols + k] * t2value.[k*t2cols + j]) [|0..(t2rows-1)|] )
        RawTensorFloat32CPU.Create(result)
        
    override t.NegT() =
        let tvalue = t.Value:?>float32[]
        let result = Array.map (~-) tvalue
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SumT() =
        let tvalue = t.Value:?>float32[]
        let result = Array.reduce (+) tvalue
        upcast RawTensorFloat32CPU([|result|], [||])
    
    override t.SumT2Dim0() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>float32[]
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> tvalue.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        upcast RawTensorFloat32CPU(result, resultShape)

    override t.TransposeT2() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>float32[]
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> tvalue.[j*tcols + i])
        RawTensorFloat32CPU.Create(result)

    override t.SqueezeT(dim) =
        let tvalue = t.Value:?>float32[]
        let result = Array.copy tvalue
        upcast RawTensorFloat32CPU(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let tvalue = t.Value:?>float32[]
        let result = Array.copy tvalue
        upcast RawTensorFloat32CPU(result, shapeUnsqueeze dim t.Shape)

    override t.ViewT(shape:int[]) =
        if shapeLength t.Shape <> shapeLength shape then invalidOp <| sprintf "Cannot view Tensor of shape %A as shape %A" t.Shape shape
        let tvalue = t.Value:?>float32[]
        let result = Array.copy tvalue
        upcast RawTensorFloat32CPU(result, shape)

    override t.SignT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map (sign >> float32)
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.FloorT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map floor
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.CeilT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map ceil
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.RoundT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map round
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AbsT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map abs
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t1.ReluT() =
        let t1value = t1.Value:?>float32[]
        let result = t1value |> Array.map (max 0.f) 
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SigmoidT() =
        let t1value = t1.Value:?>float32[]
        let result = t1value |> Array.map (fun v -> 1.f / (1.f + exp -v))
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.ExpT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map exp
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.LogT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map log
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.Log10T() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map log10
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SqrtT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map sqrt
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map sin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CosT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map cos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map tan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinhT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map sinh
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CoshT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map cosh
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanhT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map tanh
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AsinT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map asin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.AcosT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map acos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.AtanT() =
        let tvalue = t.Value:?>float32[]
        let result = tvalue |> Array.map atan
        upcast RawTensorFloat32CPU(result, t.Shape)

and RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zeros(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(Array.create (shapeLength shape) 0.f, shape)

    override __.Ones(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(Array.create (shapeLength shape) 1.f, shape)

    override __.Random(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(Array.init (shapeLength shape) (fun _ -> float32 (Random.Uniform())), shape)

    override __.RandomNormal(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(Array.init (shapeLength shape) (fun _ -> float32 (Random.Normal())), shape)

    override __.Create(value:obj) : RawTensor = 
        let array, shape = value |> flatArrayAndShape<float32>
        if notNull array then 
            upcast RawTensorFloat32CPU(array, shape)
        else 
            let array, shape = value |> flatArrayAndShape<double>
            if notNull array then 
                upcast RawTensorFloat32CPU(array |> Array.map float32, shape)
            else
                let array, shape = value |> flatArrayAndShape<int>
                if notNull array then 
                    upcast RawTensorFloat32CPU(array |> Array.map float32, shape)
                else
                    invalidArg "value" "Cannot convert value to RawTensorFloat32CPU"
    