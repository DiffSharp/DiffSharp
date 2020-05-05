namespace rec DiffSharp.Backends.None

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

[<AutoOpen>]
module internal Utils = 
    let opNotSupported (t: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type %A" t)

    let opNotSupported2 (t1: DType) (t2: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type (%A, %A)" t1 t2)

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
[<AbstractClass>]
type RawTensorCPU<'T when 'T : equality>(values: 'T[], shape: int[], dtype: DType) =
    inherit RawTensor(shape, dtype, CPU, Backend.None)

    member _.Values = values

    member internal t.IndexToFlatIndex(index:int[]) =
        indexToFlatIndex t.Shape index
    
    member internal t.FlatIndexToIndex(flatIndex:int) =
        flatIndexToIndex t.Shape flatIndex

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            // printfn "rawtensor shape %A item index %A" t.Shape index
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            let vvv = t.Values.[t.IndexToFlatIndex(index)]
            vvv
        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(index:int[]) = t.CreateLike(t.[index])
    
    override t.GetSlice(fullBounds:int[,]) =
        // if fullBounds.GetLength(0) <> t.Dim then failwithf "Expecting %i-by-3 fullBounds" t.Dim
        // printfn "rfullBounds\n%A" fullBounds
        let shape =
            [|for i=0 to (fullBounds.GetLength(0) - 1) do
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

    override t.ToValues() =
        match t.Dim with
        | 0 -> box values.[0]
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override _.StackTs(tensors, dim) =
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
        t.CreateLike(result)

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
            let result = t.ZerosLike(t.Shape) :?> RawTensorCPU<'T>
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
            let result = t.ZerosLike(dilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
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
            let result = t.ZerosLike(undilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
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
            RawTensor.Create(t.ToValues(), dtype=dtype, backend=t.Backend, device=t.Device)


// Defines the math-dependent operations for `RawTensorCPU<T>` types
// using generic inline code. Each implementing type (e.g. RawTensorFloat32CPU) instantiates
// inlines these at concrete types.
//
// Most of the functions produce (value, shape) pairs for use in constructing an instance
// of the final implementing type.
[<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
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

    let inline CreateFromFlatArray (values: System.Array, shape: int[]) : (^T[] * int[]) = 
        match values with 
        | :? ( ^T[]) as arr -> arr, shape
        | _ -> invalidArg "value" (sprintf "Data unsuitable for RawTensorCPU of type %A" typeof< ^T >)

    let inline CompareTo(t1: RawTensorCPU< ^T >, t2: RawTensor) =
        NonStructuralComparison.compare (t1.ToScalar() :?> ^T ) (t2.ToScalar() :?> ^T )

    let inline RandomMultinomial ofInt (t: RawTensorCPU< ^T >, numSamples) : (^T[] * int[]) =
        if t.Dim < 1 || t.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" t.Shape
        if t.Dim = 1 then
            let p = t.Values |> Array.map float
            let result = Array.init numSamples (fun _ -> ofInt (DiffSharp.Util.Random.ChoiceIndex(p)))
            (result, [|numSamples|])
        else
            // TODO - this was float32 - why did this pass tests - add a test for other types which covers this branch?
            let p = t.ToArray() :?> ^T[,] |> Array2D.map float
            let d1 = p.GetLength(0)
            let result = Array.init (d1 * numSamples - 1) (fun i -> ofInt (DiffSharp.Util.Random.ChoiceIndex(p.[(i%numSamples),*])))
            (result, [| d1; numSamples |]) 

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensor) = 
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types %A and %A. t1:%A, t2:%A" t1.DType t2.DType t1 t2

    let inline Full(shape:int[], value: ^T) =
        let result = Array.create (shapeLength shape) value
        (result, shape)

    let inline AllClose(t1: RawTensorCPU< ^T >, t2:RawTensor, relativeTolerance: ^T, absoluteTolerance: ^T) =
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && arraysAllClose relativeTolerance absoluteTolerance t1.Values t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    let inline LtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (<) t1value t2value
        (result, t1.Shape)

    let inline GtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (>) t1value t2value
        (result, t1.Shape)

    let inline LeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (<=) t1value t2value
        (result, t1.Shape)

    let inline GeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values
        let result = Array.map2 (>=) t1value t2value
        (result, t1.Shape)

    let inline IsInfT(isinf: ^T -> bool, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map isinf t1value
        (result, t1.Shape)

    let inline IsNaNT(isnan : ^T -> bool, t1: RawTensorCPU< ^T >) =
        let t1value = t1.Values
        let result = Array.map isnan t1value
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

    let inline internal AddTTSlice(plus, t1: RawTensorCPU< ^T >, location:int[], t2: RawTensor) : (^T[] * int[]) =
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
                    result.[t1FlatIndex] <- plus result.[t1FlatIndex] t2.[globalCoords]
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
        checkCanConv1d t1.DType t2.DType t1.Shape t2.Shape stride padding 1
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputLength = t1.Shape.[2]
        let outputChannels = t2.Shape.[0]
        let kernelLength = t2.Shape.[2]
        let outputLength = int (floor (float (inputLength + 2*padding - kernelLength)/(float stride))) + 1
        let outputShape = [|batchSize; outputChannels; outputLength|]
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let t1 =
            if padding = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorCPU< ^T >
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v=0 to outputLength-1 do
                    let mutable value = zero
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelLength-1 do
                            value <- value + t2.[k, c, u] * t1.[n, c, (v*stride) + u]
                    result.[[|n; k; v|]] <- value
        result

    let inline Conv2D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
        // t1: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
        // t2: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
        checkCanConv2d t1.DType t2.DType t1.Shape t2.Shape stride padding [|1;1|]
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputHeight = t1.Shape.[2]
        let inputWidth = t1.Shape.[3]
        let outputChannels = t2.Shape.[0]
        let kernelHeight = t2.Shape.[2]
        let kernelWidth = t2.Shape.[3]
        let outputHeight = int (floor (float (inputHeight + 2*padding.[0] - kernelHeight)/(float stride.[0]))) + 1
        let outputWidth = int (floor (float (inputWidth + 2*padding.[1] - kernelWidth)/(float stride.[1]))) + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU< ^T>
        let t1 =
            if padding.[0] = 0 && padding.[1] = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding.[0] * 2
                tshape.[3] <- t1.Shape.[3] + padding.[1] * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding.[0]; padding.[1]|], t1) :?> RawTensorCPU< ^T >
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable value = zero
                        for c=0 to inputChannels-1 do
                            for u0=0 to kernelHeight-1 do
                                for u1=0 to kernelWidth-1 do
                                    value <- value + t2.[k, c, u0, u1] * t1.[n, c, (v0*stride.[0])+u0, (v1*stride.[1])+u1]
                        result.[[|n; k; v0; v1|]] <- value
        result

    let inline Conv3D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
        // t1: input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth)
        // t2: filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth)
        checkCanConv3d t1.DType t2.DType t1.Shape t2.Shape stride padding [|1;1;1|]
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputDepth = t1.Shape.[2]
        let inputHeight = t1.Shape.[3]
        let inputWidth = t1.Shape.[4]
        let outputChannels = t2.Shape.[0]
        let kernelDepth = t2.Shape.[2]
        let kernelHeight = t2.Shape.[3]
        let kernelWidth = t2.Shape.[4]
        let outputDepth = int (floor (float (inputDepth + 2*padding.[0] - kernelDepth)/(float stride.[0]))) + 1
        let outputHeight = int (floor (float (inputHeight + 2*padding.[1] - kernelHeight)/(float stride.[1]))) + 1
        let outputWidth = int (floor (float (inputWidth + 2*padding.[2] - kernelWidth)/(float stride.[2]))) + 1
        let outputShape = [|batchSize; outputChannels; outputDepth; outputHeight; outputWidth|]
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU< ^T>
        let t1 =
            if padding.[0] = 0 && padding.[1] = 0 && padding.[2] = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding.[0] * 2
                tshape.[3] <- t1.Shape.[3] + padding.[1] * 2
                tshape.[4] <- t1.Shape.[4] + padding.[2] * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding.[0]; padding.[1]; padding.[2]|], t1) :?> RawTensorCPU< ^T >
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputDepth-1 do
                    for v1=0 to outputHeight-1 do
                        for v2=0 to outputWidth-1 do
                            let mutable value = zero
                            for c=0 to inputChannels-1 do
                                for u0=0 to kernelDepth-1 do
                                    for u1=0 to kernelHeight-1 do
                                        for u2=0 to kernelWidth-1 do
                                            // printfn "%A %A %A | %A %A %A" v0 v1 v2 u0 u1 u2
                                            value <- value + t2.[k, c, u0, u1, u2] * t1.[n, c, (v0*stride.[0])+u0, (v1*stride.[1])+u1, (v2*stride.[2])+u2]
                            result.[[|n; k; v0; v1; v2|]] <- value
        result

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
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Uniform()))
        (values, shape)

    let inline RandomNormal ofDouble (shape:int[]) : (^T[] * int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Normal()))
        (values, shape)

/// The concrete implementation of RawTensor for Float32 data.
type RawTensorFloat32CPU(values: float32[], shape:int[]) =
    inherit RawTensorCPU<float32>(values, shape, Float32)
    static let create(values, shape) : RawTensor = upcast RawTensorFloat32CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat32CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial float32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, float32 relativeTolerance, float32 absoluteTolerance)
    override t.IsInfT() = RawTensorCPU.IsInfT(System.Single.IsInfinity, t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT(System.Single.IsNaN, t) |> createBool
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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
type RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorFloat32CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorFloat32CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorFloat32CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorFloat32CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToSingle value) |> RawTensorFloat32CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random float32 shape |> RawTensorFloat32CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal float32 shape |> RawTensorFloat32CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorFloat32CPU)

type RawTensorFloat64CPU(values: double[], shape:int[]) =
    inherit RawTensorCPU<double>(values, shape, Float64)

    static let create(values, shape) : RawTensor = upcast RawTensorFloat64CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial double (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, relativeTolerance, absoluteTolerance)
    override t.IsInfT() = RawTensorCPU.IsInfT(System.Double.IsInfinity, t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT(System.Double.IsNaN, t) |> createBool
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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

type RawTensorFloat64CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorFloat64CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorFloat64CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorFloat64CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorFloat64CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToDouble value) |> RawTensorFloat64CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random double shape |> RawTensorFloat64CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal double shape |> RawTensorFloat64CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorFloat64CPU)

type RawTensorInt8CPU(values: int8[], shape:int[]) =
    inherit RawTensorCPU<int8>(values, shape, Int8)

    static let create(values, shape) : RawTensor = upcast RawTensorInt8CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt8CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt8CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int8 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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

type RawTensorInt8CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt8CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt8CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt8CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt8CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToSByte value) |> RawTensorInt8CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int8 shape |> RawTensorInt8CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int8 shape |> RawTensorInt8CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt8CPU)

type RawTensorInt16CPU(values: int16[], shape:int[]) =
    inherit RawTensorCPU<int16>(values, shape, Int16)

    static let create(values, shape) : RawTensor = upcast RawTensorInt16CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt16CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt16CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int16 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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

type RawTensorInt16CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt16CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt16CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt16CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt16CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt16 value) |> RawTensorInt16CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int16 shape |> RawTensorInt16CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int16 shape |> RawTensorInt16CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt16CPU)

type RawTensorInt32CPU(values: int32[], shape:int[]) =
    inherit RawTensorCPU<int32>(values, shape, Int32)

    static let create(values, shape) : RawTensor = upcast RawTensorInt32CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt32CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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

type RawTensorInt32CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt32CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt32CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt32CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt32CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt32 value) |> RawTensorInt32CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int32 shape |> RawTensorInt32CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int32 shape |> RawTensorInt32CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt32CPU)
                
type RawTensorInt64CPU(values: int64[], shape:int[]) =
    inherit RawTensorCPU<int64>(values, shape, Int64)

    static let create(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt64CPU(values, shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int64 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
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
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
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

type RawTensorInt64CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt64CPU)
    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt64CPU)
    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt64CPU)
    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt64CPU)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt64 value) |> RawTensorInt64CPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int64 shape |> RawTensorInt64CPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int64 shape |> RawTensorInt64CPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt64CPU)

type RawTensorBoolCPU(values: bool[], shape:int[]) =
    inherit RawTensorCPU<bool>(values, shape, Bool)

    static let create(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)
    static let create64(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
       
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorBoolCPU))
    override t.CreateShaped(values, shape) = upcast RawTensorBoolCPU(values, shape)
    override t.RandomMultinomial(_numSamples) = opNotSupported t.DType
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t1.LtTT(t2) = RawTensorBoolCPU(Array.map2 (<) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.GtTT(t2) = RawTensorBoolCPU(Array.map2 (>) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.LeTT(t2) = RawTensorBoolCPU(Array.map2 (<=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.GeTT(t2) = RawTensorBoolCPU(Array.map2 (>=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = RawTensorBoolCPU(Array.map2 (||) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.AddTT0(t2) = t1.AddTT(t2.Expand(t1.Shape))
    override t1.AddT2T1(t2) = t1.AddTT(t2.Expand(t1.Shape))
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((||), t1, location, t2) |> create
    override t1.MulTT(t2) = RawTensorBoolCPU(Array.map2 (&&) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
    override t1.MulTT0(t2) = t1.MulTT(t2.Expand(t1.Shape))
    override t.SumT() = RawTensorCPU.SumT(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
    override t.SignT() = t :> _
    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
    override t.IsNaNT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create

    override t1.SubTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.SubT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.SubTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t1.DivTT(t2) = opNotSupported2 t1.DType t2.DType
    override t1.DivT0T(t2) = opNotSupported2 t1.DType t2.DType
    override t1.DivTT0(t2) = opNotSupported2 t1.DType t2.DType
    override t1.MatMulT2T2(t2) = opNotSupported2 t1.DType t2.DType
    override t1.Conv1D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
    override t1.Conv2D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
    override t1.Conv3D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
    override t.NegT() = opNotSupported t.DType
    override t.AbsT() = opNotSupported t.DType
    override t.ReluT() = opNotSupported t.DType
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

type RawTensorBoolCPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero = upcast  RawTensorBoolCPU([| false |], [||])
    override _.One = upcast RawTensorBoolCPU([| true |], [||])
    override _.Zeros(shape:int[]) = upcast RawTensorBoolCPU(Array.zeroCreate (shapeLength shape), shape)
    override _.Ones(shape:int[]) = upcast RawTensorBoolCPU(Array.create (shapeLength shape) true, shape)
    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToBoolean value) |> RawTensorBoolCPU)
    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random (fun x -> x > 0.5) shape |> RawTensorBoolCPU)
    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.Random (fun x -> x > 0.5) shape |> RawTensorBoolCPU)
    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorBoolCPU)

