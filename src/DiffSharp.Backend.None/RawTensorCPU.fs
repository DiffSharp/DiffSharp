namespace DiffSharp.Backend.None

open DiffSharp.Backend
open DiffSharp.Util

type RawTensorFloat32CPU(values: float32[], shape:int[]) =
    inherit RawTensor(shape, Float32, CPU, Backend.None)

    member __.Values = values

    member private t.IndexToFlatIndex(index:int[]) =
        indexToFlatIndex t.Shape index
    
    member private t.FlatIndexToIndex(flatIndex:int) =
        flatIndexToIndex t.Shape flatIndex

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)]
        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(index:int[]) = upcast RawTensorFloat32CPU.Create(t.[index])
    
    override t.GetSlice(fullBounds:int[,]) =
        // if fullBounds.GetLength(0) <> t.Dim then failwithf "Expecting %i-by-3 fullBounds" t.Dim
        // printfn "rfullBounds\n%A" fullBounds
        let mutable shape = [|for i=0 to (fullBounds.GetLength(0) - 1) do
                                let len = fullBounds.[i,1] - fullBounds.[i,0] + 1
                                if fullBounds.[i, 2] = 1 then
                                    if len > 1 then yield len // if len=1 then squeeze this dimension
                                else
                                    yield len|]
        // printfn "rshape\n%A" shape
        let array = Array.create (shapeLength shape) 0.f
        let mutable arrayi = 0
        let rec slice (fullBounds:int[,]) externalCoords =
            if fullBounds.GetLength(0) = 1 then
                for i=fullBounds.[0,0] to fullBounds.[0,1] do
                    // printfn "inner %A" i
                    let globalCoords = Array.append externalCoords [|i|]
                    array.[arrayi] <- t.[globalCoords]
                    arrayi <- arrayi + 1
            else
                for i=fullBounds.[0,0] to fullBounds.[0,1] do
                    // printfn "outer %A" i
                    slice fullBounds.[1..,*] (Array.append externalCoords [|i|])
        slice fullBounds [||]
        upcast RawTensorFloat32CPU(array, shape)

    override t1.CompareTo(t2) =
        compare (t1.ToScalar():?>float32) (t2.ToScalar():?>float32)
    
    override t.Clone() = upcast RawTensorFloat32CPU(Array.copy t.Values, Array.copy t.Shape)
    override t.CreateFromScalar(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPU([|value|], [||])
        | _ -> upcast RawTensorFloat32CPU(Array.create (shapeLength shape) value, shape)

    override t.Create(values) = upcast RawTensorFloat32CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat32CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPU.One()
    override t.Ones(shape) = upcast RawTensorFloat32CPU.Ones(shape)
    override t.Full(shape, value) = upcast RawTensorFloat32CPU.Full(shape, value)
    override t.Random(shape) = upcast RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat32CPU.RandomNormal(shape)

    override probs.RandomMultinomial(numSamples) =
        if probs.Dim < 1 || probs.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" probs.Shape
        if probs.Dim = 1 then
            let p = probs.Values |> Array.map float
            let result = Array.init numSamples (fun _ -> float32 (Random.ChoiceIndex(p)))
            upcast RawTensorFloat32CPU(result, [|numSamples|])
        else
            let p = probs.ToArray() :?> float32[,] |> Array2D.map float
            let result = Array2D.init (p.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(p.[i,*]))
            upcast RawTensorFloat32CPU.Create(result)

    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        match t.Dim with
        | 0 -> sprintf "%f" t.Values.[0]
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(sprintf "%f" (t.[globalCoords])) |> ignore
                        prefix <- ", "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    let prefix2 = sprintf ", %s%s" (String.replicate (max 1 (shape.Length-1)) "\n") (String.replicate (externalCoords.Length+1) " ")
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- prefix2
                    sb.Append("]") |> ignore
            print t.Shape [||]
            sb.ToString()

    override x.ComputeHash() = hash shape + hash values

    override t.ToScalar() =
        match t.Dim with
        | 0 -> upcast t.Values.[0]
        | _ -> failwithf "Cannot convert %Ad Tensor to scalar" t.Dim
    
    override t.Expand(newShape) =
        if shape = newShape then t :> _ else
        checkCanExpandShape shape newShape
        let trim = newShape.Length - shape.Length
        let exp = shapeLength newShape.[0..trim-1]
        let jshape = newShape.[trim..]
        let n = shapeLength newShape
        let result = Array.zeroCreate n 
        if jshape.Length = 0 then 
            // The expansion is everything
            for jP = 0 to exp-1 do
                result.[jP] <- values.[0]
        else
            for jP = 0 to exp-1 do
                let rec loop ibase jbase d = 
                    let strideD = if (shape.[d] = jshape.[d]) then 1 else 0
                    if d < jshape.Length-1 then
                        let mutable iD = 0
                        for jD = 0 to jshape.[d]-1 do 
                            let ibaseD = (ibase+iD)*shape.[d+1]
                            let jbaseD = (jbase+jD)*jshape.[d+1]
                            loop ibaseD jbaseD (d+1)
                            iD <- iD + strideD
                    else
                        let mutable iD = 0
                        // last loop does the actual copy fragments
                        for jD = 0 to jshape.[d]-1 do 
                            result.[jbase+jD] <- values.[ibase+iD]
                            iD <- iD + strideD
                loop 0 (jP*jshape.[0]) 0
        RawTensorFloat32CPU(result, newShape) :> _

    override t.ToArray() =
        match t.Dim with
        | 0 -> failwith "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override t1.Equals(t2:RawTensor) = 
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        let relativeTolerance = float32 <| relativeTolerance
        let absoluteTolerance = float32 <| absoluteTolerance
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && arraysAllClose relativeTolerance absoluteTolerance t1.Values t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override __.StackTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorFloat32CPU).Values, t.Shape) |> Array.unzip
        checkCanStack shapes
        let shape = shapes.[0]
        if dim < 0 || dim > shape.Length then invalidArg "dim" "invalid dimension"
        let n = tensors |> Array.length
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim..]
        let m1 = shapeLength shape1
        let m2 = shapeLength shape2
        let m = m1 * m2
        let result = Array.zeroCreate (n * m)
        for i=0 to (n*m)-1 do
            let chunk = i/m2
            let i2 = chunk%n
            let j2 = (chunk/n)*m2+i%m2
            result.[i] <-values.[i2].[j2]

        let outShape = [| yield! shape1; yield n; yield! shape2 |]
        upcast RawTensorFloat32CPU(result, outShape)

    override t.UnstackT(dim) =
        checkCanUnstack t.Dim
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape = t.Shape
        let n = shape.[dim]
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let m1 = shapeLength shape1
        let m2 = shapeLength shape2
        let unstackedShape = Array.append shape1 shape2
        let m = m1 * m2
        let values = t.Values
        let results = Array.init n (fun _ -> Array.zeroCreate m)
        for i=0 to (n*m)-1 do
            let chunk = i/m2
            let i2 = chunk%n
            let j2 = (chunk/n)*m2+i%m2
            results.[i2].[j2] <- values.[i]
        results |> Array.map (fun rvalues -> upcast RawTensorFloat32CPU(rvalues, unstackedShape))

    override __.CatTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorFloat32CPU).Values, t.Shape) |> Array.unzip
        let n = shapes.Length
        if n = 0 then invalidArg "tensors" "Expecting at least one tensor"
        let shape = shapes.[0]
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        if shapes |> Array.exists (fun shapeOther -> shapeOther.[0..dim-1] <> shape1 || shapeOther.[dim+1..] <> shape2) then
            invalidArg "tensors" "Expecting Tensors with similar shapes"
        let m1 = shapeLength shape1
        let m2 = shapes |> Array.sumBy (fun shape -> shape.[dim])
        let m3 = shapeLength shape2
        let m = m1 * m2 * m3
        let result = Array.zeroCreate m
        let outShape = [| yield! shape1; yield m2; yield! shape2 |]
        let mutable i = 0
        for j1 = 0 to m1-1 do 
            for k = 0 to n-1 do
                let d = shapes.[k].[dim]
                let b = j1*m3*d
                for j2 = 0 to d*m3-1 do
                    result.[i+j2] <-values.[k].[b+j2]
                i <- i + d*m3

        upcast RawTensorFloat32CPU(result, outShape)

    override t.SplitT(sizes, dim) =
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape = t.Shape
        if Array.sum sizes <> shape.[dim] then invalidArg "sizes" "the sum of sizes must equal the relevant dimension"
        let n = sizes.Length
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let m1 = shapeLength shape1
        let m3 = shapeLength shape2
        let values = t.Values
        let results = Array.init n (fun k -> Array.zeroCreate (m1 * sizes.[k] * m3))
        let mutable i = 0
        for j1 = 0 to m1-1 do 
            for k = 0 to n-1 do
                let d = sizes.[k]
                let b = j1*m3*d
                for j2 = 0 to d*m3-1 do
                    results.[k].[b+j2] <-values.[i+j2]
                i <- i + d*m3

        results |> Array.mapi (fun k rvalues -> 
            let splitShape = [| yield! shape1; yield sizes.[k]; yield! shape2 |]
            upcast RawTensorFloat32CPU(rvalues, splitShape))

    override t.TransposeT2() =
        checkCanTranspose t.Dim
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> t.Values.[j*tcols + i])
        upcast RawTensorFloat32CPU.Create(result)

    override t.SqueezeT(dim) =
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) =
        checkCanFlip t.Dim dims
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(t.Shape)
            let rec flip (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[mirrorCoordinates globalCoords t.Shape dims] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        flip shape.[1..] (Array.append externalCoords [|i|])
            flip t.Shape [||]        
            upcast result

    override t.DilateT(dilations:int[]) =
        checkCanDilate t.Dim dilations
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(dilatedShape t.Shape dilations)
            let rec dilate (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[dilatedCoordinates globalCoords dilations] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        dilate shape.[1..] (Array.append externalCoords [|i|])
            dilate t.Shape [||]        
            upcast result        

    override t.UndilateT(dilations:int[]) =
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(undilatedShape t.Shape dilations)
            let rec dilate (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[globalCoords] <- t.[dilatedCoordinates globalCoords dilations]
                else
                    for i=0 to shape.[0]-1 do
                        dilate shape.[1..] (Array.append externalCoords [|i|])
            dilate result.Shape [||]        
            upcast result        

    override t.ViewT(shape:int[]) =
        checkCanView t.Shape shape
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shape)

    static member Zero() =
        let values = [|0.f|]
        RawTensorFloat32CPU(values, [||])

    static member One() =
        let values = [|1.f|]
        RawTensorFloat32CPU(values, [||])
    
    static member Zeros(shape:int[]) : RawTensorFloat32CPU =
        let values = Array.create (shapeLength shape) 0.f
        RawTensorFloat32CPU(values, shape)

    static member Ones(shape:int[]) =
        let values = Array.create (shapeLength shape) 1.f
        RawTensorFloat32CPU(values, shape)

    static member Full(shape:int[], value:obj) =
        let value = System.Convert.ToSingle(value)
        let values = Array.create (shapeLength shape) value
        RawTensorFloat32CPU(values, shape)

    static member Random(shape:int[])  =
        let values = Array.init (shapeLength shape) (fun _ -> float32 (Random.Uniform()))
        RawTensorFloat32CPU(values, shape)

    static member RandomNormal(shape:int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> float32 (Random.Normal()))
        RawTensorFloat32CPU(values, shape)

    static member Create(value:obj) = 
        let array, shape = value |> flatArrayAndShape<float32>
        if notNull array then 
            RawTensorFloat32CPU(array, shape)
        else 
            let array, shape = value |> flatArrayAndShape<double>
            if notNull array then 
                RawTensorFloat32CPU(array |> Array.map float32, shape)
            else
                let array, shape = value |> flatArrayAndShape<int>
                if notNull array then 
                    RawTensorFloat32CPU(array |> Array.map float32, shape)
                else
                    failwithf "Cannot convert value of type %A to RawTensorFloat32CPU" (value.GetType())

    override t1.LtTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 < t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GtTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 > t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.LeTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 <= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GeTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 >= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.MaxIndexT() =
        t.FlatIndexToIndex(maxIndex t.Values)

    override t.MinIndexT() =
        t.FlatIndexToIndex(minIndex t.Values)

    override t1.AddTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (+) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map ((+) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddT2T1(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTTSlice(location:int[], t2) =
        checkCanAddSlice t1.Shape location t2.Shape
        let t1value = t1.Values
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
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (-) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map ((-) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.SubTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t - t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (*) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map ((*) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (/) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map ((/) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.DivTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t / t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 ( ** ) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map (fun t -> t1value ** t) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.PowTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MatMulT2T2(t2) =
        checkCanMatmul t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values        
        let result = Array2D.init t1rows t2cols (fun i j -> Array.sumBy (fun k -> t1value.[i*t1cols + k] * t2value.[k*t2cols + j]) [|0..(t2rows-1)|] )
        upcast RawTensorFloat32CPU.Create(result)
    
    override t1.Conv1D(t2, stride, padding) =
        // t1: input, NxCxI (batchSize x inputChannels x inputLength)
        // t2: filters, KxCxF (outputChannels x inputChannels x kernelLength)
        checkCanConv1d t1.Shape t2.Shape stride padding 1
        let t1 =
            if padding = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding * 2
                let t = RawTensorFloat32CPU.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorFloat32CPU
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputLength = t1.Shape.[2]
        let outputChannels = t2.Shape.[0]
        let kernelLength = t2.Shape.[2]
        let outputLength = inputLength - kernelLength + 1
        let outputShape = [|batchSize; outputChannels; outputLength|]
        let result = RawTensorFloat32CPU.Zeros(outputShape)
        let t2 = t2 :?> RawTensorFloat32CPU
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v=0 to outputLength-1 do
                    let mutable value = 0.f
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelLength-1 do
                            value <- value + t2.[k, c, u] * t1.[n, c, v + u]
                    result.[[|n; k; v|]] <- value
        if stride = 1 then
            result :> RawTensor
        else
            let outputLength = (float outputLength) / (float stride) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputLength|]
            let mutable sresult = RawTensorFloat32CPU.Zeros(outputShape)
            for v=0 to outputLength-1 do
                let sliceBounds = array2D [[0; batchSize-1; 1]; [0; outputChannels-1; 1]; [v * stride; v * stride; 1]]
                let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1|])
                sresult <- sresult.AddTTSlice([|0; 0; v|], slice) :?> RawTensorFloat32CPU
            sresult :> RawTensor

    override t1.Conv2D(t2, stride, padding) =
        // t1: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
        // t2: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
        checkCanConv2d t1.Shape t2.Shape stride padding [|1;1|]
        let t1 =
            if padding.[0] = 0 && padding.[1] = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding.[0] * 2
                tshape.[3] <- t1.Shape.[3] + padding.[1] * 2
                let t = RawTensorFloat32CPU.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding.[0]; padding.[1]|], t1) :?> RawTensorFloat32CPU
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputHeight = t1.Shape.[2]
        let inputWidth = t1.Shape.[3]
        let outputChannels = t2.Shape.[0]
        let kernelHeight = t2.Shape.[2]
        let kernelWidth = t2.Shape.[3]
        let outputHeight = inputHeight - kernelHeight + 1
        let outputWidth = inputWidth - kernelWidth + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        let result = RawTensorFloat32CPU.Zeros(outputShape)
        let t2 = t2 :?> RawTensorFloat32CPU
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable value = 0.f
                        for c=0 to inputChannels-1 do
                            for u0=0 to kernelHeight-1 do
                                for u1=0 to kernelWidth-1 do
                                    value <- value + t2.[k, c, u0, u1] * t1.[n, c, v0+u0, v1+u1]
                        result.[[|n; k; v0; v1|]] <- value
        if stride.[0] = 1 && stride.[1] = 1 then
            result :> RawTensor
        else
            let outputHeight = (float outputHeight) / (float stride.[0]) |> ceil |> int
            let outputWidth = (float outputWidth) / (float stride.[1]) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
            let mutable sresult = RawTensorFloat32CPU.Zeros(outputShape)
            for v0=0 to outputHeight-1 do
                for v1=0 to outputWidth-1 do
                    let sliceBounds = array2D [[0; batchSize-1; 1]; [0; outputChannels-1; 1]; [v0 * stride.[0]; v0 * stride.[0]; 1]; [v1 * stride.[1]; v1 * stride.[1]; 1];]
                    let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1; 1|])
                    sresult <- sresult.AddTTSlice([|0; 0; v0; v1|], slice) :?> RawTensorFloat32CPU
            sresult :> RawTensor

    override t.NegT() =
        let result = Array.map (~-) t.Values
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SumT() =
        let result = Array.reduce (+) t.Values
        upcast RawTensorFloat32CPU([|result|], [||])
    
    override t.SumT2Dim0() =
        if t.Dim <> 2 then failwith "Expecting a 2d Tensor"
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> t.Values.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        upcast RawTensorFloat32CPU(result, resultShape)

    override t.SignT() =
        let result = t.Values |> Array.map (sign >> float32)
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.FloorT() =
        let result = t.Values |> Array.map floor
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.CeilT() =
        let result = t.Values |> Array.map ceil
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.RoundT() =
        let result = t.Values |> Array.map round
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AbsT() =
        let result = t.Values |> Array.map abs
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.ReluT() =
        let result = t.Values |> Array.map (max 0.f) 
        upcast RawTensorFloat32CPU(result, t.Shape)
        // (-a.abs()).exp().add(1.).log().add(a.max(0.))

    override t.SoftplusT() =
        let result = t.Values |> Array.map (fun x -> (max 0.f x) + log(1.f + exp(-abs(x))))
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SigmoidT() =
        let result = t.Values |> Array.map (fun v -> 1.f / (1.f + exp -v))
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.ExpT() =
        let result = t.Values |> Array.map exp
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.LogT() =
        let result = t.Values |> Array.map log
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.Log10T() =
        let result = t.Values |> Array.map log10
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SqrtT() =
        let result = t.Values |> Array.map sqrt
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinT() =
        let result = t.Values |> Array.map sin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CosT() =
        let result = t.Values |> Array.map cos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanT() =
        let result = t.Values |> Array.map tan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinhT() =
        let result = t.Values |> Array.map sinh
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CoshT() =
        let result = t.Values |> Array.map cosh
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanhT() =
        let result = t.Values |> Array.map tanh
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AsinT() =
        let result = t.Values |> Array.map asin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.AcosT() =
        let result = t.Values |> Array.map acos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.AtanT() =
        let result = t.Values |> Array.map atan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
and RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat32CPU.Zero()
    override __.One = upcast RawTensorFloat32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat32CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorFloat32CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorFloat32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat32CPU.Create(values)

 