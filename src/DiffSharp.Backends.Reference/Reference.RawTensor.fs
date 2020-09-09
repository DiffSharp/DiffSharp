#if TEST_DUPLICATE_BACKEND
namespace rec DiffSharp.Backends.TestDuplicate
#else
namespace rec DiffSharp.Backends.Reference
#endif

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

[<AutoOpen>]
module internal Utils = 
    type RawTensor with
        member x.GetTypedValues() : 'T[] = (x :?> RawTensorCPU<'T>).Values

/// This is the base class for all RawTensorXyz tuypes.
/// All type-independent operations are implemented directly on this class. 
[<AbstractClass>]
type RawTensorCPU<'T when 'T : equality>(values: 'T[], shape: int[], dtype: Dtype, device: Device) =
    inherit RawTensor()

    override _.Shape = shape
    override _.Dim = shape.Length
    override _.Nelement = shapeLength shape
    override _.Dtype = dtype
    override _.Device = device
    override _.DeviceType = device.DeviceType
    override _.Backend =
#if TEST_DUPLICATE_BACKEND
        Backend.Register "TestDuplicate"
#else
        Backend.Reference
#endif

    member _.Values : 'T[] = values

    member internal t.IndexToFlatIndex(index:int[]) =
        indexToFlatIndex t.Shape index
    
    member internal t.FlatIndexToIndex(flatIndex:int) =
        flatIndexToIndex t.Shape flatIndex

    override t.GetItem(indexes) = box t.[indexes]

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            // printfn "rawtensor shape %A item index %A" t.Shape index
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            let vvv = t.Values.[t.IndexToFlatIndex(index)]
            vvv

        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetSlice(fullBounds:int[,]) =
        // printfn "rfullBounds\n%A" fullBounds
        let shape = Shape.checkCanGetSlice t.Shape fullBounds
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
        t.MakeLike(array, shape)

    override t.Clone() = t.MakeLike(Array.copy t.Values, Array.copy t.Shape)

    abstract member MakeLike: values: 'T[] * shape: int[] * ?device: Device -> RawTensor

    override x.ComputeHash() = hash shape + hash values
    
    override t.Expand(newShape) =
        if shape = newShape then t :> _ else
        Shape.checkCanExpand shape newShape
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
        t.MakeLike(result, newShape)

    override t.ToValues() =
        match t.Dim with
        | 0 -> box values.[0]
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override _.StackTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> t.GetTypedValues(), t.Shape) |> Array.unzip
        let n, shape1, shape2, newShape = Shape.checkCanStack shapes dim
        let m1 = shapeLength shape1
        let m2 = shapeLength shape2
        let m = m1 * m2
        let result = Array.zeroCreate (n * m)
        for i=0 to (n*m)-1 do
            let chunk = i/m2
            let i2 = chunk%n
            let j2 = (chunk/n)*m2+i%m2
            result.[i] <-values.[i2].[j2]

        (tensors.[0] :?> RawTensorCPU<'T>).MakeLike(result, newShape)

    override t.UnstackT(dim) =
        let shape = t.Shape
        let shape1, shape2, unstackedShape = Shape.checkCanUnstack shape dim
        let n = shape.[dim]
        let m1 = shapeLength shape1
        let m2 = shapeLength shape2
        let m = m1 * m2
        let values = t.Values
        let results = Array.init n (fun _ -> Array.zeroCreate m)
        for i=0 to (n*m)-1 do
            let chunk = i/m2
            let i2 = chunk%n
            let j2 = (chunk/n)*m2+i%m2
            results.[i2].[j2] <- values.[i]
        results |> Array.map (fun rvalues -> t.MakeLike(rvalues, unstackedShape))

    override t.CatTs(tensors, dim) =
        let values, shapes = tensors |> Array.map (fun t -> t.GetTypedValues(), t.Shape) |> Array.unzip
        let n, shape1, m2, shape3, outShape = Shape.checkCanCat shapes dim
        let m1 = shapeLength shape1
        let m3 = shapeLength shape3
        let m = m1 * m2 * m3
        let result = Array.zeroCreate m
        let mutable i = 0
        for j1 = 0 to m1-1 do 
            for k = 0 to n-1 do
                let d = shapes.[k].[dim]
                let b = j1*m3*d
                for j2 = 0 to d*m3-1 do
                    result.[i+j2] <-values.[k].[b+j2]
                i <- i + d*m3

        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
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

        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, outShape))

    override t.TransposeT(dim0, dim1) =
        Shape.checkCanTranspose t.Shape dim0 dim1
        if dim0 = dim1 then
            let result = Array.copy t.Values
            t.MakeLike(result, t.Shape)
        else
            let shape = Array.copy t.Shape
            shape.[dim0] <- t.Shape.[dim1]
            shape.[dim1] <- t.Shape.[dim0]
            let result = t.ZerosLike(shape) :?> RawTensorCPU<'T>
            let rec transpose (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        let transposedCoords = Array.copy globalCoords
                        transposedCoords.[dim0] <- globalCoords.[dim1]
                        transposedCoords.[dim1] <- globalCoords.[dim0]
                        result.[transposedCoords] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        transpose shape.[1..] (Array.append externalCoords [|i|])
            transpose t.Shape [||]        
            upcast result

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> t.Values.[j*tcols + i])
        t.CreateLike(result)

    override t.SqueezeT(dim) =
        let result = Array.copy t.Values
        t.MakeLike(result, Shape.squeeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        let result = Array.copy t.Values
        t.MakeLike(result, outputShape)

    override t.FlipT(dims:int[]) =
        Shape.checkCanFlip t.Dim dims
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
        Shape.checkCanDilate t.Dim dilations
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = t.ZerosLike(Shape.dilated t.Shape dilations) :?> RawTensorCPU<'T>
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
            let result = t.ZerosLike(Shape.undilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
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

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype
        let indices = indices :?> RawTensorCPU<int>
        let result = t.ZerosLike(indices.Shape) :?> RawTensorCPU<'T>
        let rec gather (shape:int[]) externalCoords =
            if shape.Length = 1 then
                for i=0 to shape.[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let globalCoordsIndices = Array.copy globalCoords
                    globalCoordsIndices.[dim] <- indices.[globalCoords]
                    result.[globalCoords] <- t.[globalCoordsIndices]
            else
                for i=0 to shape.[0]-1 do
                    gather shape.[1..] (Array.append externalCoords [|i|])
        gather result.Shape [||]
        upcast result

    override t.ViewT(shape:int[]) =
        Shape.checkCanView t.Shape shape
        let result = Array.copy t.Values
        t.MakeLike(result, shape)

    override t.Cast(dtype: Dtype) =
        if dtype = t.Dtype then 
            upcast t
        else
            let tflat = t.ViewT([|t.Nelement|]) // We flatten, cast, and return with the correct shape because .ToValues() in the next line does not support tensors with dimension > 4.
            RawTensor.Create(tflat.ToValues(), dtype=dtype, backend=t.Backend, device=t.Device).ViewT(t.Shape)

    override t.MoveTo(device: Device) = t.MakeLike(values, shape, device=device)

// Defines the math-dependent operations for `RawTensorCPU<T>` types
// using generic inline code. Each implementing type (e.g. RawTensorFloat32) instantiates
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

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensor) = 
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> invalidOp <| sprintf "Cannot compare RawTensors t1 (Shape=%A, Dtype=%A, Device=%A, Backend=%A) and t2 (Shape=%A, Dtype=%A, Device=%A, Backend=%A)" t1.Shape t1.Dtype t1.Device t1.Backend t2.Shape t2.Dtype t2.Device t2.Backend

    let inline Full(shape:int[], value: ^T) =
        let result = Array.create (shapeLength shape) value
        (result, shape)

    let inline AllClose(t1: RawTensorCPU< ^T >, t2:RawTensor, relativeTolerance: ^T, absoluteTolerance: ^T) =
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && Array.allClose relativeTolerance absoluteTolerance t1.Values t2.Values
        | _ -> invalidOp <| sprintf "Cannot compare RawTensors t1 (Shape=%A, Dtype=%A, Device=%A, Backend=%A) and t2 (Shape=%A, Dtype=%A, Device=%A, Backend=%A)" t1.Shape t1.Dtype t1.Device t1.Backend t2.Shape t2.Dtype t2.Device t2.Backend

    let inline ClampT(t: RawTensorCPU< ^T>, low: RawTensor, high:RawTensor) : (^T[] * int[]) =
        if low.Dim <> 0 || high.Dim <> 0 then failwithf "Expecting scalar low and high"
        let tvalue = t.Values
        let lowvalue = low.GetTypedValues().[0]
        let highvalue = high.GetTypedValues().[0]
        let result = Array.map (fun v -> (max (min v highvalue) lowvalue)) tvalue
        (result, t.Shape)

    let inline LtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<) t1value t2value
        (result, t1.Shape)

    let inline GtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (>) t1value t2value
        (result, t1.Shape)

    let inline LeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<=) t1value t2value
        (result, t1.Shape)

    let inline GeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (>=) t1value t2value
        (result, t1.Shape)

    let inline EqTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (=) t1value t2value
        (result, t1.Shape)

    let inline NeqTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<>) t1value t2value
        (result, t1.Shape)

    let inline MaxIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(Seq.maxIndex t.Values)

    let inline MinIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(Seq.minIndex t.Values)

    let inline AddTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (+) t1value t2value
        (result, t1.Shape)

    let inline AddTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues().[0]
        let result = Array.map ((+) t2value) t1value
        (result, t1.Shape)

    let inline AddT2T1(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        (result, t1.Shape)

    let inline internal AddTTSlice(plus, t1: RawTensorCPU< ^T >, location:int[], t2: RawTensor) : (^T[] * int[]) =
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let t1value = t1.Values
        let t2 = t2 :?> RawTensorCPU< ^T >
        let result = Array.copy t1value
        let shape2 = Shape.unsqueezeAs t2.Shape t1.Shape
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
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (-) t1value t2value
        (result, t1.Shape)

    let inline SubT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = t2.GetTypedValues()
        let result = Array.map ((-) t1value) t2value
        (result, t2.Shape)

    let inline SubTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues().[0]
        let result = Array.map (fun t -> t - t2value) t1value
        (result, t1.Shape)

    let inline MulTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (*) t1value t2value
        (result, t1.Shape)

    let inline MulTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues().[0]
        let result = Array.map ((*) t2value) t1value
        (result, t1.Shape)

    let inline DivTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (/) t1value t2value
        (result, t1.Shape)

    let inline DivT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = t2.GetTypedValues()
        let result = Array.map ((/) t1value) t2value
        (result, t2.Shape)

    let inline DivTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues().[0]
        let result = Array.map (fun t -> t / t2value) t1value
        (result, t1.Shape)

    let inline PowTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 ( ** ) t1value t2value
        (result, t1.Shape)

    let inline PowT0T(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values.[0]
        let t2value = t2.GetTypedValues()
        let result = Array.map (fun t -> t1value ** t) t2value
        (result, t2.Shape)

    let inline PowTT0(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues().[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        (result, t1.Shape)

    let inline MatMulTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * int[]) =
        let (t1BatchPart, t1MatrixPart), (t2BatchPart, t2MatrixPart) = Shape.checkCanMatmul t1.Shape t2.Shape
        if t1BatchPart <> t2BatchPart then failwithf "Cannot matrix multiply raw tensors with shapes %A, %A - mismatch batching" t1.Shape t2.Shape
        let t1rows, t1cols = t1MatrixPart.[0], t1MatrixPart.[1]
        let t2rows, t2cols = t2MatrixPart.[0], t2MatrixPart.[1]
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values        
        let newShape = Array.append t1BatchPart [| t1rows; t2cols |]
        let values = 
            match t1.Dim with 
            | 2 ->
                Array.initFlat2D t1rows t2cols (fun i j -> Array.sumBy (fun k -> t1value.[i*t1cols + k] * t2value.[k*t2cols + j]) [|0..(t2rows-1)|] )
            | 3 ->
                let nb = t1BatchPart.[0]
                Array.initFlat3D nb t1rows t2cols (fun b i j -> Array.sumBy (fun k -> t1value.[b*t1cols*t1rows + i*t1cols + k] * t2value.[b*t2cols*t2rows + k*t2cols + j]) [|0..(t2rows-1)|] )
            | 4 ->
                let nb0 = t1BatchPart.[0]
                let nb1 = t1BatchPart.[1]
                Array.initFlat4D nb0 nb1 t1rows t2cols (fun b0 b1 i j -> Array.sumBy (fun k -> t1value.[((b0*nb1+b1)*t1rows+i)*t1cols+k] * t2value.[((b0*nb1+b1)*t2rows+k)*t2cols+j]) [|0..(t2rows-1)|] )
            | _ -> failwith "MatMulTT - tensor size > 4 nyi"

        (values, newShape)
    
    let inline MaxPool1D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let batchSize, channels, inputSize, outputSize, outputShape =
            Shape.checkCanMaxpool1d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1.[t1.MinIndexT()] - one
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v=0 to outputSize-1 do
                    let mutable maxvalue = minValue
                    let mutable maxindex = -1
                    for u=0 to kernelSize-1 do
                        let i = (v*stride) + u - padding
                        if i >= 0 && i < inputSize then
                            let value = t1.[n, c, i]
                            if value > maxvalue then
                                maxvalue <- value
                                maxindex <- i
                    result.[[|n; c; v|]] <- maxvalue
                    indices.[[|n; c; v|]] <- maxindex
        result, indices

    let inline MaxPool2D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), outputShape =
            Shape.checkCanMaxpool2d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1.[t1.MinIndexT()] - one
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable maxvalue = minValue
                        let mutable maxindexi0 = -1
                        let mutable maxindexi1 = -1
                        for u0=0 to kernelHeight-1 do
                            for u1=0 to kernelWidth-1 do
                                let i0 = (v0*stride.[0]) + u0 - padding.[0]
                                let i1 = (v1*stride.[1]) + u1 - padding.[1]
                                if i0 >= 0 && i0 < inputHeight && i1 >= 0 && i1 < inputWidth then
                                    let value = t1.[n, c, i0, i1]
                                    if value > maxvalue then
                                        maxvalue <- value
                                        maxindexi0 <- i0
                                        maxindexi1 <- i1
                        result.[[|n; c; v0; v1|]] <- maxvalue
                        indices.[[|n; c; v0; v1|]] <- indexToFlatIndex [|inputHeight; inputWidth|] [|maxindexi0; maxindexi1|]
        result, indices

    let inline MaxPool3D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let (batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), outputShape) =
            Shape.checkCanMaxpool3d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1.[t1.MinIndexT()] - one
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputDepth-1 do
                    for v1=0 to outputHeight-1 do
                        for v2=0 to outputWidth-1 do
                            let mutable maxvalue = minValue
                            let mutable maxindexi0 = -1
                            let mutable maxindexi1 = -1
                            let mutable maxindexi2 = -1
                            for u0=0 to kernelDepth-1 do
                                for u1=0 to kernelHeight-1 do
                                    for u2=0 to kernelWidth-1 do
                                        let i0 = (v0*stride.[0]) + u0 - padding.[0]
                                        let i1 = (v1*stride.[1]) + u1 - padding.[1]
                                        let i2 = (v2*stride.[2]) + u2 - padding.[2]
                                        if i0 >= 0 && i0 < inputDepth && i1 >= 0 && i1 < inputHeight && i2 >= 0 && i2 < inputWidth then
                                            let value = t1.[n, c, i0, i1, i2]
                                            if value > maxvalue then
                                                maxvalue <- value
                                                maxindexi0 <- i0
                                                maxindexi1 <- i1
                                                maxindexi2 <- i2
                            result.[[|n; c; v0; v1; v2|]] <- maxvalue
                            indices.[[|n; c; v0; v1; v2|]] <- indexToFlatIndex [|inputDepth; inputHeight; inputWidth|] [|maxindexi0; maxindexi1; maxindexi2|]
        result, indices

    let inline MaxUnpool1D(t1: RawTensorCPU< ^T >, indices: RawTensorCPU<int>, outputSize: int[]) : RawTensorCPU< ^T > =
        let batchSize, channels, inputSize, outputShape =
            Shape.checkCanMaxunpool1d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for u=0 to inputSize-1 do
                    let i = indices.[[|n; c; u|]]
                    result.[[|n; c; i|]] <- t1.[[|n; c; u|]]
        result

    let inline MaxUnpool2D(t1: RawTensorCPU< ^T >, indices: RawTensorCPU<int>, outputSize:int[]) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputHeight, inputWidth), outputShape =
            Shape.checkCanMaxunpool2d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for u0=0 to inputHeight-1 do
                    for u1=0 to inputWidth-1 do
                        let iflat = indices.[[|n; c; u0; u1|]]
                        let i = flatIndexToIndex [|outputSize.[2]; outputSize.[3]|] iflat
                        result.[[|n; c; i.[0]; i.[1]|]] <- t1.[[|n; c; u0; u1|]]
        result

    let inline MaxUnpool3D(t1: RawTensorCPU< ^T >, indices: RawTensorCPU<int>, outputSize:int[]) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputDepth, inputHeight, inputWidth), outputShape =
            Shape.checkCanMaxunpool3d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for u0=0 to inputDepth-1 do
                    for u1=0 to inputHeight-1 do
                        for u2=0 to inputWidth-1 do
                            let iflat = indices.[[|n; c; u0; u1; u2|]]
                            let i = flatIndexToIndex [|outputSize.[2]; outputSize.[3]; outputSize.[4]|] iflat
                            result.[[|n; c; i.[0]; i.[1]; i.[2]|]] <- t1.[[|n; c; u0; u1; u2|]]
        result

    let inline Conv1D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride, padding) : RawTensorCPU< ^T > =
        // t1: input, NxCxI (batchSize x inputChannels x inputLength)
        // t2: filters, KxCxF (outputChannels x inputChannels x kernelLength)
        let batchSize, inputChannels, kernelSize, outputChannels, outputSize, outputShape =
            Shape.checkCanConv1d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding 1
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
                for v=0 to outputSize-1 do
                    let mutable value = zero
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelSize-1 do
                            value <- value + t2.[k, c, u] * t1.[n, c, (v*stride) + u]
                    result.[[|n; k; v|]] <- value
        result

    let inline Conv2D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
        // t1: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
        // t2: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
        let batchSize, inputChannels, (kernelHeight, kernelWidth), (outputChannels, outputHeight, outputWidth), outputShape =
            Shape.checkCanConv2d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1|]
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
        let batchSize, inputChannels, (kernelDepth, kernelHeight, kernelWidth), (outputChannels, outputDepth, outputHeight, outputWidth), outputShape = 
            Shape.checkCanConv3d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1;1|]  
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

    let inline NegT op (t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = Array.map op t.Values
        (result, t.Shape)

    let inline SumT(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = Array.reduce (+) t.Values
        ([|result|], [||])
    
    let inline SumT2Dim0(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> t.Values.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        (result, resultShape)

    let inline SignT op (t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map op
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

    let inline AbsT op (t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        let result = t.Values |> Array.map op
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

    let inline RandomInt ofInt (shape:int[]) (low:int) (high:int) : (^T[] * int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> ofInt (DiffSharp.Util.Random.Integer(low, high)))
        (values, shape)

[<AbstractClass>]
type ReferenceBackendStatics() = 

    inherit BackendStatics()

    override _.GetDevices(deviceType) =
        match deviceType with 
        | None -> [ Device.CPU (* ; Device.GPU *) ]
        | Some DeviceType.CPU -> [ Device.CPU]
        //| Some DeviceType.CUDA -> [ Device.GPU ]
        | Some _ -> []

    override _.IsDeviceTypeSupported (deviceType) = (match deviceType with DeviceType.CPU | DeviceType.CUDA -> true | _ -> false)

/// The concrete implementation of RawTensor for Float32 data.
type RawTensorFloat32(values: float32[], shape:int[], device) =
    inherit RawTensorCPU<float32>(values, shape, Dtype.Float32, device)
    let create(values, shape) : RawTensor = upcast RawTensorFloat32(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device) 

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorFloat32(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, float32 relativeTolerance, float32 absoluteTolerance)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> float32) t |> create
    override t.FloorT() = RawTensorCPU.FloorT(t) |> create
    override t.CeilT() = RawTensorCPU.CeilT(t) |> create
    override t.RoundT() = RawTensorCPU.RoundT(t) |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
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

/// The concrete implementation of BackendStatics for Float32 data.
#if TEST_DUPLICATE_BACKEND
type TestDuplicateFloat32Statics() = 
#else
type ReferenceFloat32Statics() = 
#endif

    inherit ReferenceBackendStatics()
    let create device (values, shape) : RawTensor = upcast RawTensorFloat32(values, shape, device)

    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToSingle value) |> create device
    override _.Random(shape:int[], device) = RawTensorCPU.Random float32 shape |> create device
    override _.RandomNormal(shape:int[], device) = RawTensorCPU.RandomNormal float32 shape |> create device
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt float32 shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorFloat64(values: double[], shape:int[], device) =
    inherit RawTensorCPU<double>(values, shape, Dtype.Float64, device)

    let create(values, shape) : RawTensor = upcast RawTensorFloat64(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorFloat64(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, relativeTolerance, absoluteTolerance)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> double) t |> create
    override t.FloorT() = RawTensorCPU.FloorT(t) |> create
    override t.CeilT() = RawTensorCPU.CeilT(t) |> create
    override t.RoundT() = RawTensorCPU.RoundT(t) |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
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

#if TEST_DUPLICATE_BACKEND
type TestDuplicateFloat64Statics() = 
#else
type ReferenceFloat64Statics() = 
#endif

    inherit ReferenceBackendStatics()
    let create device (values, shape) : RawTensor = upcast RawTensorFloat64(values, shape, device)

    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToDouble value) |> create device
    override _.Random(shape:int[], device) = RawTensorCPU.Random double shape |> create device
    override _.RandomNormal(shape:int[], device) = RawTensorCPU.RandomNormal double shape |> create device
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt double shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorInt8(values: int8[], shape:int[], device) =
    inherit RawTensorCPU<int8>(values, shape, Dtype.Int8, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt8(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorInt8(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> int8) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateInt8Statics() = 
#else
type ReferenceInt8Statics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorInt8(values, shape, device)
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToSByte value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Int8
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Int8
    override _.RandomInt(shape, low, high, device) = RawTensorCPU.RandomInt int8 shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorByte(values: byte[], shape:int[], device) =
    inherit RawTensorCPU<byte>(values, shape, Dtype.Byte, device)

    let create(values, shape) : RawTensor = upcast RawTensorByte(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorByte(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (sbyte >> (~-) >> byte ) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (min 1uy) t |> create
    override t.AbsT() = RawTensorCPU.AbsT id t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateByteStatics() = 
#else
type ReferenceByteStatics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorByte(values, shape, device)
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToByte value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Byte
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Byte
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt byte shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorInt16(values: int16[], shape:int[], device) =
    inherit RawTensorCPU<int16>(values, shape, Dtype.Int16, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt16(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorInt16(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> int16) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateInt16Statics() = 
#else
type ReferenceInt16Statics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorInt16(values, shape, device)
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToInt16 value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Int16
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Int16
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt int16 shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorInt32(values: int32[], shape:int[], device) =
    inherit RawTensorCPU<int32>(values, shape, Dtype.Int32, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt32(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorInt32(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> int32) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateInt32Statics() = 
#else
type ReferenceInt32Statics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorInt32(values, shape, device)
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToInt32 value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Int32
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Int32
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt int32 shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorInt64(values: int64[], shape:int[], device) =
    inherit RawTensorCPU<int64>(values, shape, Dtype.Int64, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt64(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorInt64(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t.ClampT(low, high) = RawTensorCPU.ClampT(t, low, high) |> create
    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> createBool
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> createBool
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
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = RawTensorCPU.SignT (sign >> int64) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateInt64Statics() = 
#else
type ReferenceInt64Statics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorInt64(values, shape, device)

    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = RawTensorCPU.Zero() |> create device
    override _.One(device) = RawTensorCPU.One() |> create device
    override _.Zeros(shape:int[], device) = RawTensorCPU.Zeros(shape) |> create device
    override _.Ones(shape:int[], device) = RawTensorCPU.Ones(shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToInt64 value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Int64
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Int64
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt int64 shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device

type RawTensorBool(values: bool[], shape:int[], device) =
    inherit RawTensorCPU<bool>(values, shape, Dtype.Bool, device)

    let create(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorBool(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t1.LtTT(t2) = t1.MakeLike(Array.map2 (<) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.GtTT(t2) = t1.MakeLike(Array.map2 (>) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.LeTT(t2) = t1.MakeLike(Array.map2 (<=) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.GeTT(t2) = t1.MakeLike(Array.map2 (>=) t1.Values (t2.GetTypedValues()), t1.Shape) 
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> create
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> create
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = t1.MakeLike(Array.map2 (||) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.AddTT0(t2) = t1.AddTT(t2.Expand(t1.Shape))
    override t1.AddT2T1(t2) = t1.AddTT(t2.Expand(t1.Shape))
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((||), t1, location, t2) |> create
    override t1.MulTT(t2) = t1.MakeLike(Array.map2 (&&) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.MulTT0(t2) = t1.MulTT(t2.Expand(t1.Shape))
    override t.SumT(resultType) = t.Cast(Int64).SumT(?resultType=resultType)
    override t.SumT2Dim0() = t.Cast(Int64).SumT2Dim0()
    override t.SignT() = t :> _

    override t.ClampT(_low, _high) = opNotSupported "Clamp" t.Dtype
    override t1.SubTT(t2) = opNotSupported2 "SubTT" t1.Dtype t2.Dtype
    override t1.SubT0T(t2) = opNotSupported2 "SubT0T" t1.Dtype t2.Dtype
    override t1.SubTT0(t2) = opNotSupported2 "SubTT0" t1.Dtype t2.Dtype
    override t1.DivTT(t2) = opNotSupported2 "DivTT" t1.Dtype t2.Dtype
    override t1.DivT0T(t2) = opNotSupported2 "DivT0T" t1.Dtype t2.Dtype
    override t1.DivTT0(t2) = opNotSupported2 "DivTT0" t1.Dtype t2.Dtype
    override t1.MatMulTT(t2) = opNotSupported2 "MatMulTT" t1.Dtype t2.Dtype
    override t1.MaxPool1D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool1D" t1.Dtype
    override t1.MaxPool2D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool2D" t1.Dtype
    override t1.MaxPool3D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool3D" t1.Dtype
    override t1.MaxUnpool1D(_indices, _outputSize) = opNotSupported "MaxUnpool1D" t1.Dtype
    override t1.MaxUnpool2D(_indices, _outputSize) = opNotSupported "MaxUnpool2D" t1.Dtype
    override t1.MaxUnpool3D(_indices, _outputSize) = opNotSupported "MaxUnpool3D" t1.Dtype
    override t1.Conv1D(t2, _stride, _padding) = opNotSupported2 "Conv1D" t1.Dtype t2.Dtype
    override t1.Conv2D(t2, _stride, _padding) = opNotSupported2 "Conv2D" t1.Dtype t2.Dtype
    override t1.Conv3D(t2, _stride, _padding) = opNotSupported2 "Conv3D" t1.Dtype t2.Dtype
    override t.NegT() = opNotSupported "NegT" t.Dtype
    override t.AbsT() = opNotSupported "AbsT" t.Dtype
    override t.ReluT() = opNotSupported "ReluT" t.Dtype
    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t1.PowT0T(t2) = opNotSupported2 "PowT0T" t1.Dtype t2.Dtype
    override t1.PowTT0(t2) = opNotSupported2 "PowTT0" t1.Dtype t2.Dtype
    override t.FloorT() = opNotSupported "FloorT" t.Dtype
    override t.CeilT() = opNotSupported "CeilT" t.Dtype
    override t.RoundT() = opNotSupported "RoundT" t.Dtype
    override t.SigmoidT() = opNotSupported "SigmoidT" t.Dtype
    override t.ExpT() = opNotSupported "ExpT" t.Dtype
    override t.LogT() = opNotSupported "LogT" t.Dtype
    override t.Log10T() = opNotSupported "Log10T" t.Dtype
    override t.SqrtT() = opNotSupported "SqrtT" t.Dtype
    override t.SinT() = opNotSupported "SinT" t.Dtype
    override t.CosT() = opNotSupported "CosT" t.Dtype
    override t.TanT() = opNotSupported "TanT" t.Dtype
    override t.SinhT() = opNotSupported "SinhT" t.Dtype
    override t.CoshT() = opNotSupported "CoshT" t.Dtype
    override t.TanhT() = opNotSupported "TanhT" t.Dtype
    override t.AsinT() = opNotSupported "AsinT" t.Dtype
    override t.AcosT() = opNotSupported "AcosT" t.Dtype
    override t.AtanT() = opNotSupported "AtanT" t.Dtype

#if TEST_DUPLICATE_BACKEND
type TestDuplicateBoolStatics() = 
#else
type ReferenceBoolStatics() = 
#endif

    inherit ReferenceBackendStatics()

    let create device (values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(device) = ([| false |], [||]) |> create device
    override _.One(device) = ([| true |], [||]) |> create device
    override _.Zeros(shape:int[], device) = (Array.zeroCreate (shapeLength shape), shape) |> create device
    override _.Ones(shape:int[], device) = (Array.create (shapeLength shape) true, shape) |> create device
    override _.Full(shape:int[], value:obj, device) = RawTensorCPU.Full (shape, System.Convert.ToBoolean value) |> create device
    override _.Random(_shape:int[], _device) = opNotSupported "Random" Dtype.Bool
    override _.RandomNormal(_shape:int[], _device) = opNotSupported "RandomNormal" Dtype.Bool
    override _.RandomInt(shape:int[], low:int, high:int, device) = RawTensorCPU.RandomInt System.Convert.ToBoolean shape low high |> create device
    override _.CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> create device
