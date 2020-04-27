namespace DiffSharp.Backend.None

open System
open System.Text
open DiffSharp.Backend
open DiffSharp.Util
open System.Diagnostics.CodeAnalysis

#nowarn "60" // override in augmentation
#nowarn "77" // use of op_Explicit

/// This is the implementation of all RawTensorCPU types.
type RawTensorCPU<'T when 'T : equality>(values: 'T[], shape: int[], dtype: DType) =
    inherit RawTensor(shape, dtype, CPU, Backend.None)
    
    new (values: 'T[], shape: int[]) = RawTensorCPU(values, shape, DType.ofType<'T>)

    member __.Values = values

    member internal t.IndexToFlatIndex(index:int[]) =
        indexToFlatIndex t.Shape index
    
    member internal t.FlatIndexToIndex(flatIndex:int) =
        flatIndexToIndex t.Shape flatIndex

    member t.Item
        with get ([<ParamArray>] index:int[]) =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)]
        and set ([<ParamArray>] index:int[]) v =
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
            let sb = StringBuilder()
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
[<ExcludeFromCodeCoverage>]
module internal RawTensorCPU = 

    /// Access the natural "0" value for the element of a CPU tensor type
    let inline zero< ^T when ^T : (static member Zero : ^T) > = LanguagePrimitives.GenericZero< ^T >

    /// Access the natural "1" value for the element of a CPU tensor type
    let inline one< ^T when ^T : (static member One : ^T) > = LanguagePrimitives.GenericOne< ^T >
    
    /// Get the scalar "0" tensor for a CPU tensor type
    let inline Zero () =
        let values = [|zero< ^T > |]
        RawTensorCPU< ^T >(values, [| |])

    /// Get the scalar "1" tensor for a CPU tensor type
    let inline One() =
        let values = [| one< ^T > |]
        RawTensorCPU< ^T >(values, [| |])
    
    /// Get the "0" tensor for a CPU tensor type of the given shape
    let inline Zeros(shape:int[]) =
        let values = Array.create (shapeLength shape) zero< ^T >
        RawTensorCPU< ^T >(values, shape)

    let inline Ones(shape:int[]) =
        let values = Array.create (shapeLength shape) one< ^T >
        RawTensorCPU< ^T >(values, shape)

    let inline Create ofFloat32 ofFloat64 ofInt8 ofInt16 ofInt32 ofInt64 ofBool (value:obj) = 
        let values, shape = value |> flatArrayAndShape<float32>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofFloat32, shape) else 
        let values, shape = value |> flatArrayAndShape<double>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofFloat64, shape) else
        let values, shape = value |> flatArrayAndShape<int32>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofInt32, shape) else
        let values, shape = value |> flatArrayAndShape<int64>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofInt64, shape) else
        let values, shape = value |> flatArrayAndShape<int8>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofInt8, shape) else
        let values, shape = value |> flatArrayAndShape<int16>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofInt16, shape) else
        let values, shape = value |> flatArrayAndShape<bool>
        if notNull values then RawTensorCPU< ^T >(values |> Array.map ofBool, shape) else
        invalidArg "value" "Cannot convert value to RawTensorCPU"

    let inline CompareTo(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        NonStructuralComparison.compare (t1.ToScalar() :?> ^T ) (t2.ToScalar() :?> ^T )

    let inline RandomMultinomial ofInt (t: RawTensorCPU< ^T >, numSamples) =
        if t.Dim < 1 || t.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" t.Shape
        if t.Dim = 1 then
            let p = t.Values |> Array.map float
            let result = Array.init numSamples (fun _ -> ofInt (Random.ChoiceIndex(p)))
            RawTensorCPU< ^T >(result, [|numSamples|])
        else
            let p = t.ToArray() :?> 'T[,] |> Array2D.map float
            let d1 = p.GetLength(0)
            let result = Array.init (d1 * numSamples - 1) (fun i -> ofInt (Random.ChoiceIndex(p.[(i%numSamples),*])))
            RawTensorCPU< ^T >(result, [| d1; numSamples |]) 

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) = 
        t1.Shape = t2.Shape && t1.Values = t2.Values
        
    let inline Full(shape:int[], value: ^T) =
        let result = Array.create (shapeLength shape) value
        RawTensorCPU< ^T >(result, shape)

    let inline AllClose(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >, relativeTolerance: ^T, absoluteTolerance: ^T) =
        t1.Shape = t2.Shape && arraysAllClose relativeTolerance absoluteTolerance t1.Values t2.Values

    let inline LtTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (fun t1 t2 -> if t1 < t2 then one else zero) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline GtTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (fun t1 t2 -> if t1 > t2 then one else zero) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline LeTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (fun t1 t2 -> if t1 <= t2 then one else zero) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline GeTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (fun t1 t2 -> if t1 >= t2 then one else zero) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline IsInfT(isinf, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map (fun t -> if isinf t then one else zero) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline IsNaNT(isnan, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map (fun t -> if isnan t then one else zero) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline MaxIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(maxIndex t.Values)

    let inline MinIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(minIndex t.Values)

    let inline AddTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (+) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline AddTT0(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values.[0]
        let result = Array.map ((+) t2value) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline AddT2T1(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline internal AddTTSlice(t1: RawTensorCPU< ^T >, location:int[], t2: RawTensorCPU< ^T >) =
        checkCanAddSlice t1.Shape location t2.Shape
        let t1value = t1.Values
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
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline SubTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (-) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline SubT0T(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values.[0]
        let t2value = t2.Values
        let result = Array.map ((-) t1value) t2value
        RawTensorCPU< ^T >(result, t2.Shape)

    let inline SubTT0(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values.[0]
        let result = Array.map (fun t -> t - t2value) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline MulTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (*) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline MulTT0(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values.[0]
        let result = Array.map ((*) t2value) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline DivTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 (/) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline DivT0T(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values.[0]
        let t2value = t2.Values
        let result = Array.map ((/) t1value) t2value
        RawTensorCPU< ^T >(result, t2.Shape)

    let inline DivTT0(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values.[0]
        let result = Array.map (fun t -> t / t2value) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline PowTT(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values
        let result = Array.map2 ( ** ) t1value t2value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline PowT0T(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values.[0]
        let t2value = t2.Values
        let result = Array.map (fun t -> t1value ** t) t2value
        RawTensorCPU< ^T >(result, t2.Shape)

    let inline PowTT0(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let t2value = t2.Values.[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        RawTensorCPU< ^T >(result, t1.Shape)

    let inline MatMulT2T2(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >) =
        checkCanMatmul t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        let t1value = t1.Values
        let t2value = t2.Values        
        let result = Array.zeroCreate (t1rows*t2cols) 
        for i in 0 .. t1rows - 1 do
            for j in 0 .. t2cols - 1 do
                let mutable acc = zero
                for k in 0..t2rows-1 do 
                    acc <- acc + t1value.[i*t1cols + k] * t2value.[k*t2cols + j]
                result.[i*t2cols + j] <- acc
        RawTensorCPU< ^T >(result,[| t1rows; t2cols |])
    
    let inline Conv1D(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >, stride, padding) : RawTensorCPU< ^T > =
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

    let inline Conv2D(t1: RawTensorCPU< ^T >, t2: RawTensorCPU< ^T >, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
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

    let inline NegT(t: RawTensorCPU< ^T >) =
        let result = Array.map (~-) t.Values
        RawTensorCPU< ^T >(result, t.Shape)

    let inline SumT(t: RawTensorCPU< ^T >) =
        let result = Array.reduce (+) t.Values
        RawTensorCPU< ^T >([|result|], [||])
    
    let inline SumT2Dim0(t: RawTensorCPU< ^T >) =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> t.Values.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        RawTensorCPU< ^T >(result, resultShape)

    let inline SignT ofInt (t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map (sign >> ofInt)
        RawTensorCPU< ^T >(result, t.Shape)

    let inline FloorT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map floor
        RawTensorCPU< ^T >(result, t.Shape)

    let inline CeilT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map ceil
        RawTensorCPU< ^T >(result, t.Shape)

    let inline RoundT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map round
        RawTensorCPU< ^T >(result, t.Shape)

    let inline AbsT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map abs
        RawTensorCPU< ^T >(result, t.Shape)

    let inline ReluT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map (max zero< ^T >) 
        RawTensorCPU< ^T >(result, t.Shape)

    let inline SoftplusT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map (fun x -> (max zero< ^T > x) + log(one< ^T > + exp(-abs(x))))
        RawTensorCPU< ^T >(result, t.Shape)

    let inline SigmoidT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map (fun v -> one / (one + exp -v))
        RawTensorCPU< ^T >(result, t.Shape)

    let inline ExpT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map exp
        RawTensorCPU< ^T >(result, t.Shape)

    let inline LogT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map log
        RawTensorCPU< ^T >(result, t.Shape)

    let inline Log10T(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map log10
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline SqrtT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map sqrt
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline SinT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map sin
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline CosT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map cos
        RawTensorCPU< ^T >(result, t.Shape)                
        
    let inline TanT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map tan
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline SinhT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map sinh
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline CoshT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map cosh
        RawTensorCPU< ^T >(result, t.Shape)                
        
    let inline TanhT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map tanh
        RawTensorCPU< ^T >(result, t.Shape)

    let inline AsinT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map asin
        RawTensorCPU< ^T >(result, t.Shape)
        
    let inline AcosT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map acos
        RawTensorCPU< ^T >(result, t.Shape)                
        
    let inline AtanT(t: RawTensorCPU< ^T >) =
        let result = t.Values |> Array.map atan
        RawTensorCPU< ^T >(result, t.Shape)

    let inline Random ofDouble (shape:int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (Random.Uniform()))
        RawTensorCPU< ^T >(values, shape)

    let inline RandomNormal ofDouble (shape:int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Normal()))
        RawTensorCPU< ^T >(values, shape)

[<AutoOpen>]
module internal InOutProof =
    let inline input<'T, 'TMid when 'T: equality and 'TMid : equality> (t: RawTensorCPU<'T>) : RawTensorCPU<'TMid> = unbox t

    let inline output<'T, 'TMid when 'T: equality and 'TMid : equality> (t: RawTensorCPU<'TMid>) : RawTensorCPU<'T> = unbox t

    let inline outputAndDowngrade<'T, 'TMid when 'T: equality and 'TMid : equality> (t: RawTensorCPU<'TMid>) : RawTensor = output<'T, 'TMid>(t) :> _

    let inline upgrade<'T when 'T : equality> (t2: RawTensor) =
        match t2 with
        | :? RawTensorCPU<'T> as t2 -> t2
        | _ -> invalidOp (sprintf "this operation not supported on tensor of type %A " (t2.GetType()))

/// The math-dependent code with internal type instantiations
type RawTensorCPU<'T when 'T : equality> with

    static member Zero() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Zero() |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Zero() |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Zero() |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Zero() |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Zero() |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Zero() |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>([| false |], [| |]) |> output<'T, bool>
        else badtype<'T, _>

    static member One() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.One() |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.One() |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.One() |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.One() |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.One() |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.One() |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>([| true |], [| |]) |> output<'T, bool>
        else badtype<'T, _>

    static member Zeros(shape:int[]) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Zeros(shape) |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Zeros(shape) |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Zeros(shape) |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Zeros(shape) |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Zeros(shape) |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Zeros(shape) |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.zeroCreate (shapeLength shape), shape)  |> output<'T, bool>
        else badtype<'T, _>

    static member Ones(shape:int[]) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Ones(shape) |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Ones(shape) |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Ones(shape) |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Ones(shape) |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Ones(shape) |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Ones(shape) |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.create (shapeLength shape) true, shape)  |> output<'T, bool>
        else badtype<'T, _>

    static member Full(shape:int[], value: obj)  =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Full(shape, Convert.ToDouble value) |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Full(shape, Convert.ToSingle value) |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Full(shape, Convert.ToSByte value) |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Full(shape, Convert.ToInt16 value) |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Full(shape, Convert.ToInt32 value) |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Full(shape, Convert.ToInt64 value) |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.create (shapeLength shape) (Convert.ToBoolean value), shape)  |> output<'T, bool>
        else badtype<'T, _>

    static member Random(shape:int[]) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Random double shape |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Random single shape |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Random int8 shape |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Random int16 shape |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Random int32 shape |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Random int64 shape |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU.Random (fun x -> x > 0.5) shape |> output<'T, bool>
        else badtype<'T, _>

    static member RandomNormal(shape:int[]) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.RandomNormal double shape |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.RandomNormal single shape |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.RandomNormal int8 shape |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.RandomNormal int16 shape |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.RandomNormal int32 shape |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.RandomNormal int64 shape |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU.Random (fun x -> x > 0.5) shape |> output<'T, bool>
        else badtype<'T, _>

    static member Create(value:obj) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Create double double double double double double (fun x -> if x then 1.0 else 0.0) (value) |> output<'T, double>
        elif istype<'T, single> then RawTensorCPU.Create single single single single single single (fun x -> if x then 1.0f else 0.0f) (value) |> output<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Create int8 int8 int8 int8 int8 int8 (fun x -> if x then 1y else 0y) (value) |> output<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Create int16 int16 int16 int16 int16 int16 (fun x -> if x then 1s else 0s) (value) |> output<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Create int32 int32 int32 int32 int32 int32 (fun x -> if x then 1 else 0) (value) |> output<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Create int64 int64 int64 int64 int64 int64 (fun x -> if x then 1L else 0L) (value) |> output<'T, int64>
        elif istype<'T, bool> then RawTensorCPU.Create (fun i -> abs i >= 1.0f) (fun i -> abs i >= 1.0) (fun i -> i <> 0y) (fun i -> i <> 0s) (fun i -> i <> 0) (fun i -> i <> 0L) id (value) |> output<'T, bool>
        else badtype<'T, _>

    override t.CreateShaped(values, shape) = upcast RawTensorCPU<'T>(values, shape)

    override t.Create(values) = upcast RawTensorCPU<'T>.Create(values)

    override t.Zero() = upcast RawTensorCPU<'T>.Zero()

    override t.Zeros(shape) = upcast RawTensorCPU<'T>.Zeros(shape)

    override t.One() = upcast RawTensorCPU<'T>.One() 

    override t.Ones(shape) = upcast RawTensorCPU<'T>.Ones(shape)

    override t.Random(shape) = upcast RawTensorCPU<'T>.Random(shape)

    override t.RandomNormal(shape) = upcast RawTensorCPU<'T>.RandomNormal(shape)

    override t.Full(shape, value) = upcast RawTensorCPU<'T>.Full(shape, value)

    override t1.CompareTo(t2) =
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.CompareTo(input<'T, double> t1, input<'T, double> t2)
        elif istype<'T, single> then RawTensorCPU.CompareTo(input<'T, single> t1, input<'T, single> t2)
        elif istype<'T, int8> then RawTensorCPU.CompareTo(input<'T, int8> t1, input<'T, int8> t2)
        elif istype<'T, int16> then RawTensorCPU.CompareTo(input<'T, int16> t1, input<'T, int16> t2)
        elif istype<'T, int32> then RawTensorCPU.CompareTo(input<'T, int32> t1, input<'T, int32> t2)
        elif istype<'T, int64> then RawTensorCPU.CompareTo(input<'T, int64> t1, input<'T, int64> t2)
        elif istype<'T, bool> then RawTensorCPU.CompareTo(input<'T, bool> t1, input<'T, bool> t2)
        else badtype<'T, _>

    override t1.Equals(t2: RawTensor) =
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.Equals(input<'T, double> t1, input<'T, double> t2)
        elif istype<'T, single> then RawTensorCPU.Equals(input<'T, single> t1, input<'T, single> t2)
        elif istype<'T, int8> then RawTensorCPU.Equals(input<'T, int8> t1, input<'T, int8> t2)
        elif istype<'T, int16> then RawTensorCPU.Equals(input<'T, int16> t1, input<'T, int16> t2)
        elif istype<'T, int32> then RawTensorCPU.Equals(input<'T, int32> t1, input<'T, int32> t2)
        elif istype<'T, int64> then RawTensorCPU.Equals(input<'T, int64> t1, input<'T, int64> t2)
        elif istype<'T, bool> then RawTensorCPU.Equals(input<'T, bool> t1, input<'T, bool> t2)
        else badtype<'T, _>

    override t.RandomMultinomial(numSamples) =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.RandomMultinomial double (input t, numSamples) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.RandomMultinomial single (input t, numSamples) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.RandomMultinomial int8 (input t, numSamples) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.RandomMultinomial int16 (input t, numSamples) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.RandomMultinomial int32 (input t, numSamples) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.RandomMultinomial int64 (input t, numSamples) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported t.DType
        else badtype<'T, _>

    override t1.AllClose(t2, relativeTolerance, absoluteTolerance) =
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.AllClose(input t1, input t2, double relativeTolerance, double absoluteTolerance)
        elif istype<'T, single> then RawTensorCPU.AllClose(input t1, input t2, single relativeTolerance, single absoluteTolerance)
        elif istype<'T, int8> then RawTensorCPU.Equals(input<'T, int8> t1, input t2)
        elif istype<'T, int16> then RawTensorCPU.Equals(input<'T, int16> t1, input t2)
        elif istype<'T, int32> then RawTensorCPU.Equals(input<'T, int32> t1, input t2)
        elif istype<'T, int64> then RawTensorCPU.Equals(input<'T, int64> t1, input t2)
        elif istype<'T, bool> then RawTensorCPU.Equals(input<'T, bool> t1, input t2)
        else badtype<'T, _>

    override t.IsInfT() =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.IsInfT(Double.IsInfinity, input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.IsInfT(Single.IsInfinity, input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.IsInfT((fun _ -> false), input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.IsInfT((fun _ -> false), input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.IsInfT((fun _ -> false), input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.IsInfT((fun _ -> false), input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported t.DType
        else badtype<'T, _>

    override t.IsNaNT() =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.IsNaNT(Double.IsNaN, input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.IsNaNT(Single.IsNaN, input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.IsNaNT((fun _ -> false), input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.IsNaNT((fun _ -> false), input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.IsNaNT((fun _ -> false), input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.IsNaNT((fun _ -> false), input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported t.DType
        else badtype<'T, _>

    override t.SoftplusT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SoftplusT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SoftplusT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int32
        elif istype<'T, int16> then opNotSupported DType.Int32
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported t.DType
        else badtype<'T, _>

    override t1.LtTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.LtTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.map2 (<) (input<'T, bool> t1).Values (input<'T, bool> t2).Values, t1.Shape)  |> outputAndDowngrade<'T, bool>
        else badtype<'T, _>

    override t1.GtTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.GtTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.map2 (>) (input<'T, bool> t1).Values (input<'T, bool> t2).Values, t1.Shape)  |> outputAndDowngrade<'T, bool>
        else badtype<'T, _>

    override t1.LeTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.LeTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.map2 (<=) (input<'T, bool> t1).Values (input<'T, bool> t2).Values, t1.Shape)  |> outputAndDowngrade<'T, bool>
        else badtype<'T, _>

    override t1.GeTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.GeTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then RawTensorCPU<bool>(Array.map2 (>=) (input<'T, bool> t1).Values (input<'T, bool> t2).Values, t1.Shape)  |> outputAndDowngrade<'T, bool>
        else badtype<'T, _>

    override t.MaxIndexT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.MaxIndexT(input<'T, double> t)
        elif istype<'T, single> then RawTensorCPU.MaxIndexT(input<'T, single> t)
        elif istype<'T, int8> then RawTensorCPU.MaxIndexT(input<'T, int8> t)
        elif istype<'T, int16> then RawTensorCPU.MaxIndexT(input<'T, int16> t)
        elif istype<'T, int32> then RawTensorCPU.MaxIndexT(input<'T, int32> t)
        elif istype<'T, int64> then RawTensorCPU.MaxIndexT(input<'T, int64> t)
        elif istype<'T, bool> then RawTensorCPU.MaxIndexT(input<'T, bool> t)
        else badtype<'T, _>

    override t.MinIndexT() =
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.MinIndexT(input<'T, double> t)
        elif istype<'T, single> then RawTensorCPU.MinIndexT(input<'T, single> t)
        elif istype<'T, int8> then RawTensorCPU.MinIndexT(input<'T, int8> t)
        elif istype<'T, int16> then RawTensorCPU.MinIndexT(input<'T, int16> t)
        elif istype<'T, int32> then RawTensorCPU.MinIndexT(input<'T, int32> t)
        elif istype<'T, int64> then RawTensorCPU.MinIndexT(input<'T, int64> t)
        elif istype<'T, bool> then RawTensorCPU.MinIndexT(input<'T, bool> t)
        else badtype<'T, _>

    override t1.AddTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.AddTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).AddTT(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.AddTT0(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.AddTT0(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).AddTT0(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.AddT2T1(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.AddT2T1(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).AddT2T1(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.AddTTSlice(location:int[], t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.AddTTSlice(input t1, location, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).AddTTSlice(location, t2.Cast(Int64))
        else badtype<'T, _>

    override t1.SubTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SubTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).SubTT(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.SubT0T(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SubT0T(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).SubT0T(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.SubTT0(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SubTT0(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).SubTT0(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.MulTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.MulTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).MulTT(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.MulTT0(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.MulTT0(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).MulTT0(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.DivTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.DivTT(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).DivTT(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.DivT0T(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.DivT0T(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).DivT0T(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.DivTT0(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.DivTT0(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).DivTT0(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.PowTT(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.PowTT(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.PowTT(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t1.PowT0T(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.PowT0T(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.PowT0T(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t1.PowTT0(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.PowTT0(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.PowTT0(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t1.MatMulT2T2(t2) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.MatMulT2T2(input t1, input t2) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t1.Cast(Int64).MatMulT2T2(t2.Cast(Int64))
        else badtype<'T, _>

    override t1.Conv1D(t2, stride, padding) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Conv1D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t1.Conv2D(t2, stride, padding) = 
        // instantiate the generic code at each supported type
        let t2 = upgrade<'T> t2
        if istype<'T, double> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.Conv2D (input t1, input t2, stride, padding) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.NegT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.NegT(input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t.Cast(Int64).NegT()
        else badtype<'T, _>

    override t.SumT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SumT(input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t.Cast(Int64).SumT()
        else badtype<'T, _>

    override t.SumT2Dim0() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SumT2Dim0(input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t.Cast(Int64).SumT2Dim0()
        else badtype<'T, _>

    override t.SignT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SignT double (input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SignT single (input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.SignT int8 (input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.SignT int16 (input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.SignT int32 (input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.SignT int64 (input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t :> _
        else badtype<'T, _>

    override t.FloorT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.FloorT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.FloorT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.CeilT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.CeilT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.CeilT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.RoundT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.RoundT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.RoundT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.AbsT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.AbsT(input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then t :> _
        else badtype<'T, _>

    override t.ReluT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, int8>
        elif istype<'T, int16> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, int16>
        elif istype<'T, int32> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, int32>
        elif istype<'T, int64> then RawTensorCPU.ReluT(input t) |> outputAndDowngrade<'T, int64>
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.SigmoidT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SigmoidT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SigmoidT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.ExpT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.ExpT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.ExpT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.LogT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.LogT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.LogT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.Log10T() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.Log10T(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.Log10T(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.SqrtT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SqrtT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SqrtT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.SinT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SinT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SinT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.CosT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.CosT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.CosT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.TanT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.TanT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.TanT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.SinhT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.SinhT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.SinhT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.CoshT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.CoshT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.CoshT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.TanhT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.TanhT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.TanhT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.AsinT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.AsinT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AsinT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.AcosT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.AcosT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AcosT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

    override t.AtanT() = 
        // instantiate the generic code at each supported type
        if istype<'T, double> then RawTensorCPU.AtanT(input t) |> outputAndDowngrade<'T, double>
        elif istype<'T, single> then RawTensorCPU.AtanT(input t) |> outputAndDowngrade<'T, single>
        elif istype<'T, int8> then opNotSupported DType.Int8
        elif istype<'T, int16> then opNotSupported DType.Int16
        elif istype<'T, int32> then opNotSupported DType.Int32
        elif istype<'T, int64> then opNotSupported DType.Int64
        elif istype<'T, bool> then opNotSupported DType.Bool
        else badtype<'T, _>

/// The concrete implementation of RawTensorStatics 
and RawTensorCPUStatics<'T when 'T : equality>() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorCPU<'T>.Zero()
    override __.One = upcast RawTensorCPU<'T>.One()
    override __.Zeros(shape:int[]) = upcast RawTensorCPU<'T>.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorCPU<'T>.Ones(shape)
    override __.Full(shape:int[], value:obj) = upcast RawTensorCPU<'T>.Full(shape, value)
    override __.Random(shape:int[]) = upcast RawTensorCPU<'T>.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorCPU<'T>.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorCPU<'T>.Create(values)

type RawTensorBoolCPUStatics() =
    inherit RawTensorCPUStatics<bool>()

type RawTensorInt8CPUStatics() =
    inherit RawTensorCPUStatics<int8>()

type RawTensorInt16CPUStatics() =
    inherit RawTensorCPUStatics<int16>()

type RawTensorInt32CPUStatics() =
    inherit RawTensorCPUStatics<int32>()

type RawTensorInt64CPUStatics() =
    inherit RawTensorCPUStatics<int64>()

type RawTensorFloat32CPUStatics() =
    inherit RawTensorCPUStatics<single>()

type RawTensorFloat64CPUStatics() =
    inherit RawTensorCPUStatics<double>()
