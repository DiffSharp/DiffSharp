namespace DiffSharp.Backend.None

open DiffSharp.Backend
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
[<AbstractClass>]
type RawTensorCPU<'T when 'T : equality>(values: 'T[], shape: int[], dtype: DType) =
    inherit RawTensor(shape, dtype, CPU, Backend.None)

    member __.Values = values

    member internal t.IndexToFlatIndex(index:int[]) =
        indexToFlatIndex t.Shape index
    
    member internal t.FlatIndexToIndex(flatIndex:int) =
        flatIndexToIndex t.Shape flatIndex

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)]
        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(index:int[]) = t.Create(t.[index])
    
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
        let array = Array.zeroCreate (shapeLength shape)
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
        t.CreateShaped(array, shape)

    override t.Clone() = t.CreateShaped(Array.copy t.Values, Array.copy t.Shape)

    abstract member CreateShaped: values: 'T[] * shape: int[] -> RawTensor

    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        let printVal (x:obj) = 
           match x with 
           | :? single as v -> sprintf "%f" v
           | :? double as v -> sprintf "%f" v
           | :? int8 as v -> sprintf "%d" v
           | :? int16 as v -> sprintf "%d" v
           | :? int32 as v -> sprintf "%d" v
           | :? int64 as v -> sprintf "%d" v
           | :? bool as v -> if v then "true" else "false"
           | _ -> sprintf "%A" x

        match t.Dim with
        | 0 -> printVal t.Values.[0]
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(printVal (t.[globalCoords])) |> ignore
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
        t.CreateShaped(result, newShape)

    override t.ToArray() =
        match t.Dim with
        | 0 -> failwith "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override __.StackTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorCPU<'T>).Values, t.Shape) |> Array.unzip
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
        (tensors.[0] :?> RawTensorCPU<'T>).CreateShaped(result, outShape)

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
        results |> Array.map (fun rvalues -> t.CreateShaped(rvalues, unstackedShape))

    override t.CatTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorCPU<'T>).Values, t.Shape) |> Array.unzip
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

        t.CreateShaped(result, outShape)

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
            t.CreateShaped(rvalues, splitShape))

    override t.TransposeT2() =
        checkCanTranspose t.Dim
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> t.Values.[j*tcols + i])
        t.Create(result)

    override t.SqueezeT(dim) =
        let result = Array.copy t.Values
        t.CreateShaped(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let result = Array.copy t.Values
        t.CreateShaped(result, shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) =
        checkCanFlip t.Dim dims
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = t.Zeros(t.Shape) :?> RawTensorCPU<'T>
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
            let result = t.Zeros(dilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
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
            let result = t.Zeros(undilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
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
        t.CreateShaped(result, shape)

    override t.Cast(dtype: DType) =
        if dtype = t.DType then 
            upcast t
        else 
            RawTensor.Create(t.ToArray(), dtype=dtype, backend=t.Backend, device=t.Device)


// Defines the math-dependent operations for `RawTensorCPU<T>` types
// using generic inline code. Each implementing type (e.g. RawTensorFloat32CPU) instantiates
// inlines these at concrete types.
//
// Most of the functions produce (value, shape) pairs for use in constructing an instance
// of the final implementing type.
module internal RawTensorCPU = 

    /// Access the natural "0" value for the element of a CPU tensor type
    let inline zero< ^T when ^T : (static member Zero : ^T) > = LanguagePrimitives.GenericZero< ^T >

    /// Access the natural "1" value for the element of a CPU tensor type
    let inline one< ^T when ^T : (static member One : ^T) > = LanguagePrimitives.GenericOne< ^T >
    
    /// Get the scalar "0" tensor for a CPU tensor type
    let inline Zero () : (^T[] * int[]) =
        let values = [|zero< ^T > |]
        (values, [| |])

    /// Get the scalar "1" tensor for a CPU tensor type
    let inline One() : (^T[] * int[]) =
        let values = [| one< ^T > |]
        (values, [| |])
    
    /// Get the "0" tensor for a CPU tensor type of the given shape
    let inline Zeros(shape:int[])  : (^T[] * int[]) =
        let values = Array.create (shapeLength shape) zero< ^T >
        (values, shape)

    let inline Ones(shape:int[]) =
        let values = Array.create (shapeLength shape) one< ^T >
        (values, shape)

    let inline Create ofFloat32 ofFloat64 ofInt8 ofInt16 ofInt32 ofInt64 ofBool (value:obj) : (^T[] * int[]) = 
        let values, shape = value |> flatArrayAndShape<float32>
        if notNull values then (values |> Array.map ofFloat32, shape) else 
        let values, shape = value |> flatArrayAndShape<double>
        if notNull values then (values |> Array.map ofFloat64, shape) else
        let values, shape = value |> flatArrayAndShape<int32>
        if notNull values then (values |> Array.map ofInt32, shape) else
        let values, shape = value |> flatArrayAndShape<int64>
        if notNull values then (values |> Array.map ofInt64, shape) else
        let values, shape = value |> flatArrayAndShape<int8>
        if notNull values then (values |> Array.map ofInt8, shape) else
        let values, shape = value |> flatArrayAndShape<int16>
        if notNull values then (values |> Array.map ofInt16, shape) else
        let values, shape = value |> flatArrayAndShape<bool>
        if notNull values then (values |> Array.map ofBool, shape) else
        invalidArg "value" "Cannot convert value to RawTensorCPU"

    let inline CompareTo(t1: RawTensorCPU< ^T >, t2: RawTensor) =
        NonStructuralComparison.compare (t1.ToScalar() :?> ^T ) (t2.ToScalar() :?> ^T )

    let inline RandomMultinomial ofInt (t: RawTensorCPU< ^T >, numSamples) : (^T[] * int[]) =
        if t.Dim < 1 || t.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" t.Shape
        if t.Dim = 1 then
            let p = t.Values |> Array.map float
            let result = Array.init numSamples (fun _ -> ofInt (Random.ChoiceIndex(p)))
            (result, [|numSamples|])
        else
            let p = t.ToArray() :?> float32[,] |> Array2D.map float
            let d1 = p.GetLength(0)
            let result = Array.init (d1 * numSamples - 1) (fun i -> ofInt (Random.ChoiceIndex(p.[(i%numSamples),*])))
            (result, [| d1; numSamples |]) 

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensor) = 
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    let inline Full(shape:int[], value: ^T) =
        let result = Array.create (shapeLength shape) value
        (result, shape)

    let inline AllClose(t1: RawTensorCPU< ^T >, t2:RawTensor, relativeTolerance: ^T, absoluteTolerance: ^T) =
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && arraysAllClose relativeTolerance absoluteTolerance t1.Values t2.Values
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    let inline LtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (fun t1 t2 -> if t1 < t2 then one else zero) t1value t2value
        (result, t1.Shape)

    let inline GtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (fun t1 t2 -> if t1 > t2 then one else zero) t1value t2value
        (result, t1.Shape)

    let inline LeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (fun t1 t2 -> if t1 <= t2 then one else zero) t1value t2value
        (result, t1.Shape)

    let inline GeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (fun t1 t2 -> if t1 >= t2 then one else zero) t1value t2value
        (result, t1.Shape)

    let inline IsInfT(isinf, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map (fun t -> if isinf t then one else zero) t1value
        (result, t1.Shape)

    let inline IsNaNT(isnan, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map (fun t -> if isnan t then one else zero) t1value
        (result, t1.Shape)

    let inline MaxIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(maxIndex t.Values)

    let inline MinIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(minIndex t.Values)

    let inline AddTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (+) t1value t2value
        (result, t1.Shape)

    let inline AddTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values.[0]
        let result = Array.map ((+) t2value) t1value
        (result, t1.Shape)

    let inline AddT2T1(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        (result, t1.Shape)

    let inline internal AddTTSlice(t1: RawTensorCPU< ^T >, location:int[], t2: RawTensor) : (^T[] * int[]) =
        checkCanAddSlice t1.Shape location t2.Shape
        let t1value = t1.Values
        let t2 = t2 :?> RawTensorCPU< ^T >
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
        (result, t1.Shape)

    let inline SubTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (-) t1value t2value
        (result, t1.Shape)

    let inline SubT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map ((-) t1value) t2value
        (result, t2.Shape)

    let inline SubTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values.[0]
        let result = Array.map (fun t -> t - t2value) t1value
        (result, t1.Shape)

    let inline MulTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (*) t1value t2value
        (result, t1.Shape)

    let inline MulTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values.[0]
        let result = Array.map ((*) t2value) t1value
        (result, t1.Shape)

    let inline DivTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (/) t1value t2value
        (result, t1.Shape)

    let inline DivT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map ((/) t1value) t2value
        (result, t2.Shape)

    let inline DivTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values.[0]
        let result = Array.map (fun t -> t / t2value) t1value
        (result, t1.Shape)

    let inline PowTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 ( ** ) t1value t2value
        (result, t1.Shape)

    let inline PowT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map (fun t -> t1value ** t) t2value
        (result, t2.Shape)

    let inline PowTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values.[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        (result, t1.Shape)

    let inline MatMulT2T2(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        checkCanMatmul t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values        
        let result = Array.zeroCreate (t1rows*t2cols) 
        for i in 0 .. t1rows - 1 do
            for j in 0 .. t2cols - 1 do
                let mutable acc = zero
                for k in 0..t2rows-1 do 
                    acc <- acc + t1value.[i*t1cols + k] * t2value.[k*t2cols + j]
                result.[i*t2cols + j] <- acc
        (result,[| t1rows; t2cols |])
    
    let inline Conv1D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride, padding) : RawTensorCPU< ^T > =
        // t1: input, NxCxI (batchSize x inputChannels x inputLength)
        // t2: filters, KxCxF (outputChannels x inputChannels x kernelLength)
        checkCanConv1d t1.Shape t2.Shape stride padding 1
        let t1 =
            if padding = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding * 2
                let t = t1.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorCPU< ^T >
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputLength = t1.Shape.[2]
        let outputChannels = t2.Shape.[0]
        let kernelLength = t2.Shape.[2]
        let outputLength = inputLength - kernelLength + 1
        let outputShape = [|batchSize; outputChannels; outputLength|]
        let result = t1.Zeros(outputShape) :?> RawTensorCPU<'T>
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v=0 to outputLength-1 do
                    let mutable value = zero
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelLength-1 do
                            value <- value + t2.[k, c, u] * t1.[n, c, v + u]
                    result.[[|n; k; v|]] <- value
        if stride = 1 then
            result 
        else
            let outputLength = (float outputLength) / (float stride) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputLength|]
            let mutable sresult = t1.Zeros(outputShape) :?> RawTensorCPU<_>
            for v=0 to outputLength-1 do
                let sliceBounds = array2D [[0; batchSize-1; 1]; [0; outputChannels-1; 1]; [v * stride; v * stride; 1]]
                let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1|])
                sresult <- sresult.AddTTSlice([|0; 0; v|], slice) :?> RawTensorCPU<_>
            sresult 

    let inline Conv2D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
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
                let t = t1.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding.[0]; padding.[1]|], t1) :?> RawTensorCPU< ^T >
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputHeight = t1.Shape.[2]
        let inputWidth = t1.Shape.[3]
        let outputChannels = t2.Shape.[0]
        let t2 = t2 :?> RawTensorCPU< ^T >
        let kernelHeight = t2.Shape.[2]
        let kernelWidth = t2.Shape.[3]
        let outputHeight = inputHeight - kernelHeight + 1
        let outputWidth = inputWidth - kernelWidth + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        let result = t1.Zeros(outputShape) :?> RawTensorCPU< ^T>
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable value = zero
                        for c=0 to inputChannels-1 do
                            for u0=0 to kernelHeight-1 do
                                for u1=0 to kernelWidth-1 do
                                    value <- value + t2.[k, c, u0, u1] * t1.[n, c, v0+u0, v1+u1]
                        result.[[|n; k; v0; v1|]] <- value
        if stride.[0] = 1 && stride.[1] = 1 then
            result
        else
            let outputHeight = (float outputHeight) / (float stride.[0]) |> ceil |> int
            let outputWidth = (float outputWidth) / (float stride.[1]) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
            let mutable sresult = t1.Zeros(outputShape)
            for v0=0 to outputHeight-1 do
                for v1=0 to outputWidth-1 do
                    let sliceBounds = array2D [[0; batchSize-1; 1]; [0; outputChannels-1; 1]; [v0 * stride.[0]; v0 * stride.[0]; 1]; [v1 * stride.[1]; v1 * stride.[1]; 1];]
                    let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1; 1|])
                    sresult <- sresult.AddTTSlice([|0; 0; v0; v1|], slice) 
            sresult :?> RawTensorCPU< ^T >

    let inline NegT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = Array.map (~-) t.Values
        (result, t.Shape)

    let inline SumT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = Array.reduce (+) t.Values
        ([|result|], [||])
    
    let inline SumT2Dim0(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> t.Values.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        (result, resultShape)

    let inline SignT ofInt (t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map (sign >> ofInt)
        (result, t.Shape)

    let inline FloorT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map floor
        (result, t.Shape)

    let inline CeilT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map ceil
        (result, t.Shape)

    let inline RoundT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map round
        (result, t.Shape)

    let inline AbsT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map abs
        (result, t.Shape)

    let inline ReluT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map (max zero< ^T >) 
        (result, t.Shape)

    let inline SoftplusT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map (fun x -> (max zero< ^T > x) + log(one< ^T > + exp(-abs(x))))
        (result, t.Shape)

    let inline SigmoidT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map (fun v -> one / (one + exp -v))
        (result, t.Shape)

    let inline ExpT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map exp
        (result, t.Shape)

    let inline LogT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map log
        (result, t.Shape)

    let inline Log10T(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map log10
        (result, t.Shape)
        
    let inline SqrtT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map sqrt
        (result, t.Shape)
        
    let inline SinT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map sin
        (result, t.Shape)
        
    let inline CosT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map cos
        (result, t.Shape)                
        
    let inline TanT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map tan
        (result, t.Shape)
        
    let inline SinhT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map sinh
        (result, t.Shape)
        
    let inline CoshT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map cosh
        (result, t.Shape)                
        
    let inline TanhT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map tanh
        (result, t.Shape)

    let inline AsinT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map asin
        (result, t.Shape)
        
    let inline AcosT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map acos
        (result, t.Shape)                
        
    let inline AtanT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map atan
        (result, t.Shape)

    let inline Random ofDouble (shape:int[]) : (^T[] * int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (Random.Uniform()))
        (values, shape)

    let inline RandomNormal ofDouble (shape:int[]) : (^T[] * int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Normal()))
        (values, shape)

/// The concrete implementation of RawTensor for Float32 data.
type RawTensorFloat32CPU(values: float32[], shape:int[]) =
    inherit RawTensorCPU<float32>(values, shape, Float32)
    static let create(values, shape) : RawTensor = upcast RawTensorFloat32CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorFloat32CPU
    static member One() = RawTensorCPU.One() |> RawTensorFloat32CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorFloat32CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorFloat32CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToSingle value) |> RawTensorFloat32CPU
    static member Random(shape:int[])  = RawTensorCPU.Random float32 shape |> RawTensorFloat32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal float32 shape |> RawTensorFloat32CPU
    static member Create(value:obj) = RawTensorCPU.Create float32 float32 float32 float32 float32 float32 (fun x -> if x then 1.0f else 0.0f) (value) |> RawTensorFloat32CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat32CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat32CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat32CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPU.One() 
    override t.Ones(shape) = upcast RawTensorFloat32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial float32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, float32 relativeTolerance, float32 absoluteTolerance)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToSingle value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT(System.Single.IsInfinity, t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT(System.Single.IsNaN, t) |> create
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t1.PowT0T(t2) = RawTensorCPU.PowT0T(t1, t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT float32 t |> create
    override t.FloorT() = RawTensorCPU.FloorT(t) |> create
    override t.CeilT() = RawTensorCPU.CeilT(t) |> create
    override t.RoundT() = RawTensorCPU.RoundT(t) |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create
    override t.SigmoidT() = RawTensorCPU.SigmoidT(t) |> create
    override t.ExpT() = RawTensorCPU.ExpT(t) |> create
    override t.LogT() = RawTensorCPU.LogT(t) |> create
    override t.Log10T() = RawTensorCPU.Log10T(t) |> create
    override t.SqrtT() = RawTensorCPU.SqrtT(t) |> create
    override t.SinT() = RawTensorCPU.SinT(t) |> create
    override t.CosT() = RawTensorCPU.CosT(t) |> create
    override t.TanT() = RawTensorCPU.TanT(t) |> create
    override t.SinhT() = RawTensorCPU.SinhT(t) |> create
    override t.CoshT() = RawTensorCPU.CoshT(t) |> create
    override t.TanhT() = RawTensorCPU.TanhT(t) |> create
    override t.AsinT() = RawTensorCPU.AsinT(t) |> create
    override t.AcosT() = RawTensorCPU.AcosT(t) |> create
    override t.AtanT() = RawTensorCPU.AtanT(t) |> create

/// The concrete implementation of RawTensorStatics for Float32 data.
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

type RawTensorFloat64CPU(values: double[], shape:int[]) =
    inherit RawTensorCPU<double>(values, shape, Float64)

    static let create(values, shape) : RawTensor = upcast RawTensorFloat64CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorFloat64CPU
    static member One() = RawTensorCPU.One() |> RawTensorFloat64CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorFloat64CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorFloat64CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToDouble value) |> RawTensorFloat64CPU
    static member Random(shape:int[])  = RawTensorCPU.Random double shape |> RawTensorFloat64CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal double shape |> RawTensorFloat64CPU
    static member Create(value:obj) = RawTensorCPU.Create double double double double double double (fun x -> if x then 1.0 else 0.0) (value) |> RawTensorFloat64CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat64CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat64CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat64CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat64CPU.One()
    override t.Ones(shape) = upcast RawTensorFloat64CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat64CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial double (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, relativeTolerance, absoluteTolerance)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToDouble value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT(System.Double.IsInfinity, t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT(System.Double.IsNaN, t) |> create
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t1.PowT0T(t2) = RawTensorCPU.PowT0T(t1, t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT double t |> create
    override t.FloorT() = RawTensorCPU.FloorT(t) |> create
    override t.CeilT() = RawTensorCPU.CeilT(t) |> create
    override t.RoundT() = RawTensorCPU.RoundT(t) |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create
    override t.SigmoidT() = RawTensorCPU.SigmoidT(t) |> create
    override t.ExpT() = RawTensorCPU.ExpT(t) |> create
    override t.LogT() = RawTensorCPU.LogT(t) |> create
    override t.Log10T() = RawTensorCPU.Log10T(t) |> create
    override t.SqrtT() = RawTensorCPU.SqrtT(t) |> create
    override t.SinT() = RawTensorCPU.SinT(t) |> create
    override t.CosT() = RawTensorCPU.CosT(t) |> create
    override t.TanT() = RawTensorCPU.TanT(t) |> create
    override t.SinhT() = RawTensorCPU.SinhT(t) |> create
    override t.CoshT() = RawTensorCPU.CoshT(t) |> create
    override t.TanhT() = RawTensorCPU.TanhT(t) |> create
    override t.AsinT() = RawTensorCPU.AsinT(t) |> create
    override t.AcosT() = RawTensorCPU.AcosT(t) |> create
    override t.AtanT() = RawTensorCPU.AtanT(t) |> create

and RawTensorFloat64CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat64CPU.Zero()
    override __.One = upcast RawTensorFloat64CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat64CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat64CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorFloat64CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorFloat64CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat64CPU.Create(values)

type RawTensorInt8CPU(values: int8[], shape:int[]) =
    inherit RawTensorCPU<int8>(values, shape, Int8)

    static let create(values, shape) : RawTensor = upcast RawTensorInt8CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorInt8CPU
    static member One() = RawTensorCPU.One() |> RawTensorInt8CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorInt8CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorInt8CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToSByte value) |> RawTensorInt8CPU
    static member Random(shape:int[])  = RawTensorCPU.Random int8 shape |> RawTensorInt8CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal int8 shape |> RawTensorInt8CPU
    static member Create(value:obj) = RawTensorCPU.Create int8 int8 int8 int8 int8 int8 (fun x -> if x then 1y else 0y) (value) |> RawTensorInt8CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt8CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt8CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt8CPU.Create(values)
    override t.Zero() = upcast RawTensorInt8CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt8CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt8CPU.One()
    override t.Ones(shape) = upcast RawTensorInt8CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt8CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt8CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int8 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToSByte value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT int8 t |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported t.DType
    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t.FloorT() = opNotSupported t.DType
    override t.CeilT() = opNotSupported t.DType
    override t.RoundT() = opNotSupported t.DType
    override t.SigmoidT() = opNotSupported t.DType
    override t.ExpT() = opNotSupported t.DType
    override t.LogT() = opNotSupported t.DType
    override t.Log10T() = opNotSupported t.DType
    override t.SqrtT() = opNotSupported t.DType
    override t.SinT() = opNotSupported t.DType
    override t.CosT() = opNotSupported t.DType
    override t.TanT() = opNotSupported t.DType
    override t.SinhT() = opNotSupported t.DType
    override t.CoshT() = opNotSupported t.DType
    override t.TanhT() = opNotSupported t.DType
    override t.AsinT() = opNotSupported t.DType
    override t.AcosT() = opNotSupported t.DType
    override t.AtanT() = opNotSupported t.DType

and RawTensorInt8CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt8CPU.Zero()
    override __.One = upcast RawTensorInt8CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt8CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt8CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorInt8CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorInt8CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt8CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt8CPU.Create(values)

type RawTensorInt16CPU(values: int16[], shape:int[]) =
    inherit RawTensorCPU<int16>(values, shape, Int16)

    static let create(values, shape) : RawTensor = upcast RawTensorInt16CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorInt16CPU
    static member One() = RawTensorCPU.One() |> RawTensorInt16CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorInt16CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorInt16CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToInt16 value) |> RawTensorInt16CPU
    static member Random(shape:int[])  = RawTensorCPU.Random int16 shape |> RawTensorInt16CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal int16 shape |> RawTensorInt16CPU
    static member Create(value:obj) = RawTensorCPU.Create int16 int16 int16 int16 int16 int16 (fun x -> if x then 1s else 0s) (value) |> RawTensorInt16CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt16CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt16CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt16CPU.Create(values)
    override t.Zero() = upcast RawTensorInt16CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt16CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt16CPU.One()
    override t.Ones(shape) = upcast RawTensorInt16CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt16CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt16CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int16 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToInt16 value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT int16 t |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported t.DType
    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t.FloorT() = opNotSupported t.DType
    override t.CeilT() = opNotSupported t.DType
    override t.RoundT() = opNotSupported t.DType
    override t.SigmoidT() = opNotSupported t.DType
    override t.ExpT() = opNotSupported t.DType
    override t.LogT() = opNotSupported t.DType
    override t.Log10T() = opNotSupported t.DType
    override t.SqrtT() = opNotSupported t.DType
    override t.SinT() = opNotSupported t.DType
    override t.CosT() = opNotSupported t.DType
    override t.TanT() = opNotSupported t.DType
    override t.SinhT() = opNotSupported t.DType
    override t.CoshT() = opNotSupported t.DType
    override t.TanhT() = opNotSupported t.DType
    override t.AsinT() = opNotSupported t.DType
    override t.AcosT() = opNotSupported t.DType
    override t.AtanT() = opNotSupported t.DType

and RawTensorInt16CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt16CPU.Zero()
    override __.One = upcast RawTensorInt16CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt16CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt16CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorInt16CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorInt16CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt16CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt16CPU.Create(values)

type RawTensorInt32CPU(values: int32[], shape:int[]) =
    inherit RawTensorCPU<int32>(values, shape, Int32)

    static let create(values, shape) : RawTensor = upcast RawTensorInt32CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorInt32CPU
    static member One() = RawTensorCPU.One() |> RawTensorInt32CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorInt32CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorInt32CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToInt32 value) |> RawTensorInt32CPU
    static member Random(shape:int[])  = RawTensorCPU.Random int32 shape |> RawTensorInt32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal int32 shape |> RawTensorInt32CPU
    static member Create(value:obj) = RawTensorCPU.Create int32 int32 int32 int32 int32 int32 (fun x -> if x then 1 else 0) (value) |> RawTensorInt32CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt32CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt32CPU.Create(values)
    override t.Zero() = upcast RawTensorInt32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt32CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt32CPU.One()
    override t.Ones(shape) = upcast RawTensorInt32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToInt32 value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT int32 t |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported t.DType
    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t.FloorT() = opNotSupported t.DType
    override t.CeilT() = opNotSupported t.DType
    override t.RoundT() = opNotSupported t.DType
    override t.SigmoidT() = opNotSupported t.DType
    override t.ExpT() = opNotSupported t.DType
    override t.LogT() = opNotSupported t.DType
    override t.Log10T() = opNotSupported t.DType
    override t.SqrtT() = opNotSupported t.DType
    override t.SinT() = opNotSupported t.DType
    override t.CosT() = opNotSupported t.DType
    override t.TanT() = opNotSupported t.DType
    override t.SinhT() = opNotSupported t.DType
    override t.CoshT() = opNotSupported t.DType
    override t.TanhT() = opNotSupported t.DType
    override t.AsinT() = opNotSupported t.DType
    override t.AcosT() = opNotSupported t.DType
    override t.AtanT() = opNotSupported t.DType

and RawTensorInt32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt32CPU.Zero()
    override __.One = upcast RawTensorInt32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt32CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorInt32CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorInt32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt32CPU.Create(values)
                
type RawTensorInt64CPU(values: int64[], shape:int[]) =
    inherit RawTensorCPU<int64>(values, shape, Int64)

    static let create(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorInt64CPU
    static member One() = RawTensorCPU.One() |> RawTensorInt64CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorInt64CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorInt64CPU
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToInt64 value) |> RawTensorInt64CPU
    static member Random(shape:int[])  = RawTensorCPU.Random int64 shape |> RawTensorInt64CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal int64 shape |> RawTensorInt64CPU
    static member Create(value:obj) = RawTensorCPU.Create int64 int64 int64 int64 int64 int64 (fun x -> if x then 1L else 0L) (value) |> RawTensorInt64CPU

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt64CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt64CPU.Create(values)
    override t.Zero() = upcast RawTensorInt64CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt64CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt64CPU.One()
    override t.Ones(shape) = upcast RawTensorInt64CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt64CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt64CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int64 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.Full(shape, value) = RawTensorCPU.Full(shape, System.Convert.ToInt64 value) |> create
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> create
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> create
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> create
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice(t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT int64 t |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported t.DType
    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t.FloorT() = opNotSupported t.DType
    override t.CeilT() = opNotSupported t.DType
    override t.RoundT() = opNotSupported t.DType
    override t.SigmoidT() = opNotSupported t.DType
    override t.ExpT() = opNotSupported t.DType
    override t.LogT() = opNotSupported t.DType
    override t.Log10T() = opNotSupported t.DType
    override t.SqrtT() = opNotSupported t.DType
    override t.SinT() = opNotSupported t.DType
    override t.CosT() = opNotSupported t.DType
    override t.TanT() = opNotSupported t.DType
    override t.SinhT() = opNotSupported t.DType
    override t.CoshT() = opNotSupported t.DType
    override t.TanhT() = opNotSupported t.DType
    override t.AsinT() = opNotSupported t.DType
    override t.AcosT() = opNotSupported t.DType
    override t.AtanT() = opNotSupported t.DType

and RawTensorInt64CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt64CPU.Zero()
    override __.One = upcast RawTensorInt64CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt64CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt64CPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorInt64CPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorInt64CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt64CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt64CPU.Create(values)

type RawTensorBoolCPU(values: bool[], shape:int[]) =
    inherit RawTensorCPU<bool>(values, shape, Bool)

    static let create64(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
    static member Zero() = RawTensorBoolCPU([| false |], [||])
    static member One() = RawTensorBoolCPU([| true |], [||])
    static member Zeros(shape:int[]) = RawTensorBoolCPU(Array.zeroCreate (shapeLength shape), shape)
    static member Ones(shape:int[]) = RawTensorBoolCPU(Array.create (shapeLength shape) true, shape)
    static member Full(shape:int[], value: obj)  = RawTensorCPU.Full (shape, System.Convert.ToBoolean value) |> RawTensorBoolCPU
    static member Random(shape:int[])  = RawTensorCPU.Random (fun x -> x > 0.5) shape |> RawTensorBoolCPU
    static member RandomNormal(shape:int[]) = RawTensorBoolCPU.Random(shape)
    static member Create(value:obj) = RawTensorCPU.Create (fun i -> abs i >= 1.0f) (fun i -> abs i >= 1.0) (fun i -> i <> 0y) (fun i -> i <> 0s) (fun i -> i <> 0) (fun i -> i <> 0L) id (value) |> RawTensorBoolCPU
       
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorBoolCPU))
    override t.CreateShaped(values, shape) = upcast RawTensorBoolCPU(values, shape)
    override t.Create(values) = upcast RawTensorBoolCPU.Create(values)
    override t.Zero() = upcast RawTensorBoolCPU.Zero()
    override t.Zeros(shape) = upcast RawTensorBoolCPU.Zeros(shape)
    override t.One() = upcast RawTensorBoolCPU.One()
    override t.Ones(shape) = upcast RawTensorBoolCPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorBoolCPU.Random(shape)
    override t.RandomNormal(_shape) = opNotSupported t.DType
    override t.RandomMultinomial(_numSamples) = opNotSupported t.DType
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.Full(shape, value) = RawTensorBoolCPU(RawTensorCPU.Full(shape, System.Convert.ToBoolean value)) :> _
    override t1.LtTT(t2) = RawTensorBoolCPU(Array.map2 (<) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.GtTT(t2) = RawTensorBoolCPU(Array.map2 (>) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.LeTT(t2) = RawTensorBoolCPU(Array.map2 (<=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.GeTT(t2) = RawTensorBoolCPU(Array.map2 (>=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((t1.Cast(Int64) :?> RawTensorCPU<int64>), location, t2) |> create64
    override t1.SubTT(t2) = RawTensorCPU.SubTT((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.SubT0T(t2) = RawTensorCPU.SubT0T((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.MulTT(t2) = RawTensorCPU.MulTT((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.DivTT(t2) = RawTensorCPU.DivTT((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.DivT0T(t2) = RawTensorCPU.DivT0T((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.MatMulT2T2(t2) = RawTensorCPU.MatMulT2T2((t1.Cast(Int64) :?> RawTensorCPU<int64>), t2.Cast(Int64)) |> create64
    override t1.Conv1D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
    override t1.Conv2D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
    override t.NegT() = RawTensorCPU.NegT(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
    override t.SumT() = RawTensorCPU.SumT(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
    override t.SignT() = t :> _
    override t.AbsT() = opNotSupported t.DType
    override t.ReluT() = RawTensorCPU.ReluT(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64

    override t.IsInfT() = opNotSupported t.DType
    override t.IsNaNT() = opNotSupported t.DType
    override t.SoftplusT() = opNotSupported t.DType
    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t.FloorT() = opNotSupported t.DType
    override t.CeilT() = opNotSupported t.DType
    override t.RoundT() = opNotSupported t.DType
    override t.SigmoidT() = opNotSupported t.DType
    override t.ExpT() = opNotSupported t.DType
    override t.LogT() = opNotSupported t.DType
    override t.Log10T() = opNotSupported t.DType
    override t.SqrtT() = opNotSupported t.DType
    override t.SinT() = opNotSupported t.DType
    override t.CosT() = opNotSupported t.DType
    override t.TanT() = opNotSupported t.DType
    override t.SinhT() = opNotSupported t.DType
    override t.CoshT() = opNotSupported t.DType
    override t.TanhT() = opNotSupported t.DType
    override t.AsinT() = opNotSupported t.DType
    override t.AcosT() = opNotSupported t.DType
    override t.AtanT() = opNotSupported t.DType

and RawTensorBoolCPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorBoolCPU.Zero()
    override __.One = upcast RawTensorBoolCPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorBoolCPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorBoolCPU.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorBoolCPU.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorBoolCPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorBoolCPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorBoolCPU.Create(values)

