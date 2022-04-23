// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

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

/// This is the base class for all RawTensorXyz types.
/// All type-independent operations are implemented directly on this class. 
[<AbstractClass>]
type RawTensorCPU<'T when 'T : equality and 'T :> scalar>(values: 'T[], shape: Shape, dtype: Dtype, device: Device) =
    inherit RawTensor()
    do if device.DeviceType = DeviceType.CUDA then failwithf "CUDA is not supported by the reference backend."

    let mutable values = values
    let mutable isMutable = false
    let checkMutable() = if not isMutable then failwith "The tensor cannot be mutated." 
    override _.Shape = shape
    override _.Dim = shape.Length
    override _.Nelement = shapeLength shape
    override _.Dtype = dtype
    override _.Device = device
    override _.DeviceType = device.DeviceType
    override _.Handle = box values
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

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            // printfn "rawtensor shape %A item index %A" t.Shape index
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            let vvv = t.Values[t.IndexToFlatIndex(index)]
            vvv

        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(indexes:int[]) =
        t[indexes] :> scalar

    override t.GetSlice(fullBounds:int[,]) =
        let fullBounds = Shape.completeSliceBounds t.Shape fullBounds
        let shape = Shape.checkCanGetSlice t.Shape fullBounds
        let array = Array.zeroCreate (shapeLength shape)
        let mutable arrayi = 0
        let rec slice (fullBounds:int[,]) externalCoords =
            if fullBounds.GetLength(0) = 1 then
                for i=fullBounds[0,0] to fullBounds[0,1] do
                    // printfn "inner %A" i
                    let globalCoords = Array.append externalCoords [|i|]
                    array[arrayi] <- t[globalCoords]
                    arrayi <- arrayi + 1
            else
                for i=fullBounds[0,0] to fullBounds[0,1] do
                    // printfn "outer %A" i
                    slice fullBounds[1..,*] (Array.append externalCoords [|i|])
        slice fullBounds [||]
        t.MakeLike(array, shape)

    override t.Clone() = t.MakeLike(Array.copy t.Values, Array.copy t.Shape)

    abstract member MakeLike: values: 'T[] * shape: Shape * ?device: Device -> RawTensor

    override x.ComputeHash() = hash shape + hash values
    
    override t.Expand(newShape) =
        if newShape.Length = 1 && newShape[0] = 0 then t.MakeLike([||], newShape) else  // Return zero-sized tensor if expanding to zero-sized tensor
        if shape = newShape then t :> _ else
        Shape.checkCanExpand shape newShape
        let trim = newShape.Length - shape.Length
        let exp = shapeLength newShape[0..trim-1]
        let jshape = newShape[trim..]
        let n = shapeLength newShape
        let result = Array.zeroCreate n 
        if jshape.Length = 0 then 
            // The expansion is everything
            for jP = 0 to exp-1 do
                result[jP] <- values[0]
        else
            for jP = 0 to exp-1 do
                let rec loop ibase jbase d = 
                    let strideD = if (shape[d] = jshape[d]) then 1 else 0
                    if d < jshape.Length-1 then
                        let mutable iD = 0
                        for jD = 0 to jshape[d]-1 do 
                            let ibaseD = (ibase+iD)*shape[d+1]
                            let jbaseD = (jbase+jD)*jshape[d+1]
                            loop ibaseD jbaseD (d+1)
                            iD <- iD + strideD
                    else
                        let mutable iD = 0
                        // last loop does the actual copy fragments
                        for jD = 0 to jshape[d]-1 do 
                            result[jbase+jD] <- values[ibase+iD]
                            iD <- iD + strideD
                loop 0 (jP*jshape[0]) 0
        t.MakeLike(result, newShape)

    override t.ToValues() =
        let shape = t.Shape
        match t.Dim with
        | 0 -> box values[0]
        | 1 -> upcast Array.init shape[0] (fun i -> t[i])
        | 2 -> upcast Array2D.init shape[0] shape[1] (fun i j -> t[i, j])
        | 3 -> upcast Array3D.init shape[0] shape[1] shape[2] (fun i j k -> t[i, j, k])
        | 4 -> upcast Array4D.init shape[0] shape[1] shape[2] shape[3] (fun i j k l -> t[i, j, k, l])
        | 5 -> upcast Array5D.init shape[0] shape[1] shape[2] shape[3] shape[4] (fun i j k l m -> t[i, j, k, l, m])
        | 6 -> upcast Array6D.init shape[0] shape[1] shape[2] shape[3] shape[4] shape[5] (fun i j k l m n -> t[i, j, k, l, m, n])
        | _ -> ArrayND.init shape (fun idxs -> t[idxs])

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
            result[i] <-values[i2][j2]

        (tensors[0] :?> RawTensorCPU<'T>).MakeLike(result, newShape)

    override t.UnstackT(dim) =
        let shape = t.Shape
        let shape1, shape2, unstackedShape = Shape.checkCanUnstack shape dim
        let n = shape[dim]
        let m1 = shapeLength shape1
        let m2 = shapeLength shape2
        let m = m1 * m2
        let values = t.Values
        let results = Array.init n (fun _ -> Array.zeroCreate m)
        for i=0 to (n*m)-1 do
            let chunk = i/m2
            let i2 = chunk%n
            let j2 = (chunk/n)*m2+i%m2
            results[i2][j2] <- values[i]
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
                let d = shapes[k][dim]
                let b = j1*m3*d
                for j2 = 0 to d*m3-1 do
                    result[i+j2] <-values[k][b+j2]
                i <- i + d*m3

        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        let n = sizes.Length
        let shape1 = shape[0..dim-1]
        let shape2 = shape[dim+1..]
        let m1 = shapeLength shape1
        let m3 = shapeLength shape2
        let values = t.Values
        let results = Array.init n (fun k -> Array.zeroCreate (m1 * sizes[k] * m3))
        let mutable i = 0
        for j1 = 0 to m1-1 do 
            for k = 0 to n-1 do
                let d = sizes[k]
                let b = j1*m3*d
                for j2 = 0 to d*m3-1 do
                    results[k][b+j2] <-values[i+j2]
                i <- i + d*m3

        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, outShape))

    override t.PermuteT(permutation) =
        let inversePermutation, newShape = Shape.checkCanPermute t.Shape permutation
        let result = t.ZerosLike(newShape) :?> RawTensorCPU<'T>
        let rec transpose (shape:Shape) externalCoords = 
            if shape.Length = 1 then
                for i=0 to shape[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let transposedCoords = Array.permute (fun i -> inversePermutation[i]) globalCoords
                    result[transposedCoords] <- t[globalCoords]
            else
                for i=0 to shape[0]-1 do
                    transpose shape[1..] (Array.append externalCoords [|i|])
        transpose t.Shape [||]        
        upcast result

    override t.TransposeT(dim0, dim1) =
        let permutation = [| 0 .. t.Shape.Length - 1 |]
        permutation[dim0] <- dim1
        permutation[dim1] <- dim0
        t.PermuteT(permutation)

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        let tcols = t.Shape[1]
        let result = Array2D.init t.Shape[1] t.Shape[0] (fun i j -> t.Values[j*tcols + i])
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
            let rec flip (shape:Shape) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result[mirrorCoordinates globalCoords t.Shape dims] <- t[globalCoords]
                else
                    for i=0 to shape[0]-1 do
                        flip shape[1..] (Array.append externalCoords [|i|])
            flip t.Shape [||]        
            upcast result

    override t.DilateT(dilations:int[]) =
        Shape.checkCanDilate t.Dim dilations
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = t.ZerosLike(Shape.dilated t.Shape dilations) :?> RawTensorCPU<'T>
            let rec dilate (shape:Shape) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result[dilatedCoordinates globalCoords dilations] <- t[globalCoords]
                else
                    for i=0 to shape[0]-1 do
                        dilate shape[1..] (Array.append externalCoords [|i|])
            dilate t.Shape [||]        
            upcast result        

    override t.UndilateT(dilations:int[]) =
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = t.ZerosLike(Shape.undilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
            let rec dilate (shape:Shape) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result[globalCoords] <- t[dilatedCoordinates globalCoords dilations]
                else
                    for i=0 to shape[0]-1 do
                        dilate shape[1..] (Array.append externalCoords [|i|])
            dilate result.Shape [||]
            upcast result

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype
        let indices = indices :?> RawTensorCPU<int>
        let result = t.ZerosLike(indices.Shape) :?> RawTensorCPU<'T>
        let rec gather (shape:Shape) externalCoords =
            if shape.Length = 1 then
                for i=0 to shape[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let globalCoordsIndices = Array.copy globalCoords
                    globalCoordsIndices[dim] <- indices[globalCoords]
                    result[globalCoords] <- t[globalCoordsIndices]
            else
                for i=0 to shape[0]-1 do
                    gather shape[1..] (Array.append externalCoords [|i|])
        gather result.Shape [||]
        upcast result

    override t.ScatterT(dim:int, indices, destinationShape:Shape) =
        Shape.checkCanScatter t.Shape dim indices.Shape indices.Dtype destinationShape
        let indices = indices :?> RawTensorCPU<int>
        let result = t.ZerosLike(destinationShape) :?> RawTensorCPU<'T>
        let rec scatter (shape:Shape) externalCoords =
            if shape.Length = 1 then
                for i=0 to shape[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let globalCoordsIndices = Array.copy globalCoords
                    globalCoordsIndices[dim] <- indices[globalCoords]
                    result[globalCoordsIndices] <- t[globalCoords]
            else
                for i=0 to shape[0]-1 do
                    scatter shape[1..] (Array.append externalCoords [|i|])
        scatter t.Shape [||]
        upcast result

    override t.ViewT(shape:Shape) =
        Shape.checkCanView t.Shape shape
        let result = Array.copy t.Values
        t.MakeLike(result, shape)

    override t.Cast(dtype: Dtype) =
        if dtype = t.Dtype then 
            upcast t
        else
            let tflat = t.ViewT([|t.Nelement|]) // We flatten, cast, and return with the correct shape because .ToValues() in the next line does not support tensors with dimension > 4.
            let values = 
                match t.Dtype with
                // These special cases for byte and int8 are to ensure that values don't get truncated because RawTensor.Create cannot distinguish between byte and int8
                | Dtype.Byte -> tflat.ToValues():?>byte[] |> Array.map int |> box
                | Dtype.Int8 -> tflat.ToValues():?>int8[] |> Array.map int |> box
                | _ -> tflat.ToValues()

            RawTensor.Create(values, dtype=dtype, backend=t.Backend, device=t.Device).ViewT(t.Shape)

    override t.MoveTo(device: Device) = t.MakeLike(values, shape, device=device)

    override t.SetMutable() = isMutable <- true
    override t.IsMutable = isMutable
    member t.SetValues(tmp: RawTensor) = checkMutable(); values <- (tmp :?> RawTensorCPU<'T>).Values
    override t.ClampInPlace(low, high) = t.SetValues <| t.ClampT(low, high)
    override t.LtInPlace(t2) = t.SetValues <| t.LtTT(t2)
    override t.GtInPlace(t2) = t.SetValues <| t.GtTT(t2)
    override t.LeInPlace(t2) = t.SetValues <| t.LeTT(t2)
    override t.GeInPlace(t2) = t.SetValues <| t.GeTT(t2)
    override t.EqInPlace(t2) = t.SetValues <| t.EqTT(t2)
    override t.NeqInPlace(t2) = t.SetValues <| t.NeqTT(t2)
    override t.AddInPlace(t2, alpha) = t.SetValues <| t.AddTT(t2, ?alpha=alpha)
    override t.AddScalarInPlace(t2) = t.SetValues <| t.AddTT0(t2)
    override t.AddSliceInPlace(location, t2) = t.SetValues <| t.AddTTSlice(location, t2)
    override t.SubInPlace(t2) = t.SetValues <| t.SubTT(t2)
    override t.SubScalarInPlace(t2) = t.SetValues <| t.SubTT0(t2)
    override t.MulInPlace(t2) = t.SetValues <| t.MulTT(t2)
    override t.MulScalarInPlace(t2) = t.SetValues <| t.MulTT0(t2)
    override t.DivInPlace(t2) = t.SetValues <| t.DivTT(t2)
    override t.DivScalarInPlace(t2) = t.SetValues <| t.DivTT0(t2)
    override t.PowInPlace(t2) = t.SetValues <| t.PowTT(t2)
    override t.PowScalarInPlace(t2) = t.SetValues <| t.PowTT0(t2)
    override t.MatMulInPlace(t2) = t.SetValues <| t.MatMulTT(t2)
    override t.NegInPlace() = t.SetValues <| t.NegT()
    override t.SignInPlace() = t.SetValues <| t.SignT()
    override t.FloorInPlace() = t.SetValues <| t.FloorT()
    override t.CeilInPlace() = t.SetValues <| t.CeilT()
    override t.RoundInPlace() = t.SetValues <| t.RoundT()
    override t.AbsInPlace() = t.SetValues <| t.AbsT()
    override t.ReluInPlace() = t.SetValues <| t.ReluT()
    override t.SoftplusInPlace() = t.SetValues <| t.SoftplusT()
    override t.SigmoidInPlace() = t.SetValues <| t.SigmoidT()
    override t.ExpInPlace() = t.SetValues <| t.ExpT()
    override t.LogInPlace() = t.SetValues <| t.LogT()
    override t.Log10InPlace() = t.SetValues <| t.Log10T()
    override t.SqrtInPlace() = t.SetValues <| t.SqrtT()
    override t.SinInPlace() = t.SetValues <| t.SinT()
    override t.CosInPlace() = t.SetValues <| t.CosT()
    override t.TanInPlace() = t.SetValues <| t.TanT()
    override t.SinhInPlace() = t.SetValues <| t.SinhT()
    override t.CoshInPlace() = t.SetValues <| t.CoshT()
    override t.TanhInPlace() = t.SetValues <| t.TanhT()
    override t.AsinInPlace() = t.SetValues <| t.AsinT()
    override t.AcosInPlace() = t.SetValues <| t.AcosT()
    override t.AtanInPlace() = t.SetValues <| t.AtanT()
    override t.OnesInPlace() = t.SetValues <| t.OnesLike(t.Shape)
    override t.RandomInPlace() = t.SetValues <| t.RandomLike(t.Shape) 
    override t.RandomNormalInPlace() = t.SetValues <| t.RandomNormalLike(t.Shape)
    override t.RandomIntInPlace(low, high) = t.SetValues <| t.RandomIntLike(t.Shape, low, high)
    override t.ZerosInPlace() = t.SetValues <| t.ZerosLike(t.Shape)

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
    let inline Zero () : (^T[] * Shape) =
        let values = [|zero< ^T > |]
        (values, Shape.scalar)

    /// Get the scalar "1" tensor for a CPU tensor type
    let inline One() : (^T[] * Shape) =
        let values = [| one< ^T > |]
        (values, Shape.scalar)
    
    /// Get the "0" tensor for a CPU tensor type of the given shape
    let inline Zeros(shape:Shape)  : (^T[] * Shape) =
        let values = Array.zeroCreate (shapeLength shape) 
        (values, shape)

    /// Get the "0" tensor for a CPU tensor type of the given shape
    let inline Empty(shape:Shape)  : (^T[] * Shape) = Zeros shape

    let inline Ones(shape:Shape) =
        let values = Array.create (shapeLength shape) one< ^T >
        (values, shape)

    let inline CreateFromFlatArray (values: System.Array, shape: Shape) : (^T[] * Shape) = 
        match values with 
        | :? ( ^T[]) as arr -> arr, shape
        | _ -> invalidArg "value" (sprintf "Data unsuitable for RawTensorCPU of type %A" typeof< ^T >)

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensor) = 
        if t1.Dtype <> t2.Dtype then 
            opNotSupported2 "Equals" t1.Dtype t2.Dtype
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> invalidOp <| sprintf "Cannot compare RawTensors t1 (Shape=%A, Dtype=%A, Device=%A, Backend=%A) and t2 (Shape=%A, Dtype=%A, Device=%A, Backend=%A)" t1.Shape t1.Dtype t1.Device t1.Backend t2.Shape t2.Dtype t2.Device t2.Backend

    let inline Full(shape:Shape, value: ^T) =
        let result = Array.create (shapeLength shape) value
        (result, shape)

    let inline AllClose(t1: RawTensorCPU< ^T >, t2:RawTensor, relativeTolerance: ^T, absoluteTolerance: ^T) =
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && Array.allClose relativeTolerance absoluteTolerance t1.Values t2.Values
        | _ -> invalidOp <| sprintf "Cannot compare RawTensors t1 (Shape=%A, Dtype=%A, Device=%A, Backend=%A) and t2 (Shape=%A, Dtype=%A, Device=%A, Backend=%A)" t1.Shape t1.Dtype t1.Device t1.Backend t2.Shape t2.Dtype t2.Device t2.Backend

    let inline ClampT(t: RawTensorCPU< ^T>, low: RawTensor, high:RawTensor) : (^T[] * Shape) =
        if low.Dim <> 0 || high.Dim <> 0 then failwithf "Expecting scalar low and high"
        let tvalue = t.Values
        let lowvalue = low.GetTypedValues()[0]
        let highvalue = high.GetTypedValues()[0]
        let result = Array.map (fun v -> (max (min v highvalue) lowvalue)) tvalue
        (result, t.Shape)

    let inline LtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<) t1value t2value
        (result, t1.Shape)

    let inline GtTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (>) t1value t2value
        (result, t1.Shape)

    let inline LeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<=) t1value t2value
        (result, t1.Shape)

    let inline GeTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (>=) t1value t2value
        (result, t1.Shape)

    let inline EqTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (=) t1value t2value
        (result, t1.Shape)

    let inline NeqTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (bool[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (<>) t1value t2value
        (result, t1.Shape)

    let inline MaxIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(Seq.maxIndex t.Values)

    let inline MinMaxReduceT op (t: RawTensorCPU< ^T >, dim, keepDim) : RawTensor * RawTensor =
        let newShape = Shape.checkCanMinMaxReduce dim keepDim t.Shape
        let shape = t.Shape
        let shape1 = shape[0..dim-1]
        let n = shape[dim]
        let shape2 = shape[dim+1..]
        let m1 = shapeLength shape1
        let m3 = shapeLength shape2
        let values = t.Values
        let results = Array.zeroCreate (m1 * m3)
        let indexes = Array.zeroCreate (m1 * m3)
        for j1 = 0 to m1-1 do 
            for j2 = 0 to m3-1 do
                let b = j1*m3 + j2
                for j3 = 0 to n-1 do
                    let v = values[j1*n*m3+j3*m3+j2]
                    if op v results[b] || (j3 = 0) then
                        results[b] <- v
                        indexes[b] <- j3
        let resultsT = t.MakeLike(results, newShape)
        let indexesT = t.CreateLike(indexes, dtype=Dtype.Int32).ViewT(newShape)
        resultsT, indexesT

    let inline MinIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(Seq.minIndex t.Values)

    let inline AddTT(t1: RawTensorCPU< ^T >, t2: RawTensor, alpha: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (fun a b -> a + alpha * b) t1value t2value
        (result, t1.Shape)

    let inline AddTT0(t1: RawTensorCPU< ^T >, b: ^T, alpha: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let result = Array.map (fun a -> a + alpha * b) t1value
        (result, t1.Shape)

    let inline internal AddTTSlice(plus, t1: RawTensorCPU< ^T >, location:int[], t2: RawTensor) : (^T[] * Shape) =
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let t1value = t1.Values
        let t2 = t2 :?> RawTensorCPU< ^T >
        let result = Array.copy t1value
        let shape2 = Shape.unsqueezeAs t2.Shape t1.Shape
        let rec add (shape2:Shape) externalCoords =
            if shape2.Length = 1 then
                for i=0 to shape2[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let t1Coords = Array.map2 (+) globalCoords location
                    let t1FlatIndex = t1.IndexToFlatIndex(t1Coords)
                    result[t1FlatIndex] <- plus result[t1FlatIndex] t2[globalCoords]
            else
                for i=0 to shape2[0]-1 do
                    add (shape2[1..]) (Array.append externalCoords [|i|])
        add shape2 [||]
        (result, t1.Shape)

    let inline SubTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (-) t1value t2value
        (result, t1.Shape)

    let inline SubT0T(a: ^T, t2: RawTensor) : (^T[] * Shape) =
        let t2value = t2.GetTypedValues()
        let result = Array.map (fun b -> a - b) t2value
        (result, t2.Shape)

    let inline SubTT0(t1: RawTensorCPU< ^T >, b: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let result = Array.map (fun t -> t - b) t1value
        (result, t1.Shape)

    let inline MulTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (*) t1value t2value
        (result, t1.Shape)

    let inline MulTT0(t1: RawTensorCPU< ^T >, b: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let result = Array.map (fun a -> a * b) t1value
        (result, t1.Shape)

    let inline DivTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 (/) t1value t2value
        (result, t1.Shape)

    let inline DivT0T(a: ^T, t2: RawTensor) : (^T[] * Shape) =
        let t2value = t2.GetTypedValues()
        let result = Array.map (fun b -> a / b) t2value
        (result, t2.Shape)

    let inline DivTT0(t1: RawTensorCPU< ^T >, b: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let result = Array.map (fun a -> a / b) t1value
        (result, t1.Shape)

    let inline PowTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        let t1value = t1.Values
        let t2value = t2.GetTypedValues()
        let result = Array.map2 ( ** ) t1value t2value
        (result, t1.Shape)

    let inline PowT0T(a: ^T , t2: RawTensor) : (^T[] * Shape) =
        let t2value = t2.GetTypedValues()
        let result = Array.map (fun b -> a ** b) t2value
        (result, t2.Shape)

    let inline PowTT0(t1: RawTensorCPU< ^T >, b: ^T) : (^T[] * Shape) =
        let t1value = t1.Values
        let result = Array.map (fun a -> a ** b) t1value
        (result, t1.Shape)

    let inline MatMulTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        let (t1BatchPart, t1MatrixPart), (t2BatchPart, t2MatrixPart) = Shape.checkCanMatmul t1.Shape t2.Shape
        if t1BatchPart <> t2BatchPart then failwithf "Cannot matrix multiply raw tensors with shapes %A, %A - mismatch batching" t1.Shape t2.Shape
        let t1rows, t1cols = t1MatrixPart[0], t1MatrixPart[1]
        let t2rows, t2cols = t2MatrixPart[0], t2MatrixPart[1]
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorCPU< ^T >).Values        
        let newShape = Array.append t1BatchPart [| t1rows; t2cols |]
        let nb = shapeLength t1BatchPart
        let values = Array.initFlat3D nb t1rows t2cols (fun b i j -> Array.sumBy (fun k -> t1value[b*t1cols*t1rows + i*t1cols + k] * t2value[b*t2cols*t2rows + k*t2cols + j]) [|0..(t2rows-1)|] )
        (values, newShape)
    
    let inline BMMTT(t1: RawTensorCPU< ^T >, t2: RawTensor) : (^T[] * Shape) =
        Shape.checkCanBMM t1.Shape t2.Shape |> ignore
        MatMulTT(t1, t2)

    // Returns the LU decomposition of this matrix. The return values are the LU matrix, pivot indices, and a toggle value indicating the number of row exchanges during the decomposition, which is +1 if the number of exchanges were even, -1 if odd. Source: Atilim Gunes Baydin, FsAlg, 2015, https://github.com/gbaydin/FsAlg
    let inline LUDecomposition (m: ^T[,]) =
        let rows = m.GetLength(0)
        let res = Array2D.copy m
        let perm = Array.init rows (fun i -> i)
        let mutable toggle = LanguagePrimitives.GenericOne<'T>
        for j = 0 to rows - 2 do
            let mutable colmax:'T = abs res[j, j]
            let mutable prow = j
            for i = j + 1 to rows - 1 do
                let absresij = abs res[i, j]
                if absresij > colmax then
                    colmax <- absresij
                    prow <- i
            if prow <> j then
                let tmprow = res[prow, 0..]
                res[prow, 0..] <- res[j, 0..]
                res[j, 0..] <- tmprow
                let tmp = perm[prow]
                perm[prow] <- perm[j]
                perm[j] <- tmp
                toggle <- -toggle
            for i = j + 1 to rows - 1 do
                res[i, j] <- res[i, j] / res[j, j]
                for k = j + 1 to rows - 1 do
                    res[i, k] <- res[i, k] - res[i, j] * res[j, k]
        res, perm, toggle

    // Finds an array that, when multiplied by a LU matrix `lu`, gives array `b`. Source: Atilim Gunes Baydin, FsAlg, 2015, https://github.com/gbaydin/FsAlg
    let inline matrixSolveHelper (lu:^T[,]) (b:^T[]) =
        let n = lu.GetLength 0
        let x = Array.copy b
        for i = 1 to n - 1 do
            let mutable sum = x[i]
            for j = 0 to i - 1 do
                sum <- sum - lu[i, j] * x[j]
            x[i] <- sum
        x[n - 1] <- x[n - 1] / lu[n - 1, n - 1]
        for i in (n - 2) .. -1 .. 0 do
            let mutable sum = x[i]
            for j = i + 1 to n - 1 do
                sum <- sum - lu[i, j] * x[j]
            x[i] <- sum / lu[i, i]
        x

    // Solves a system of linear equations ax = b, where the coefficients are given in matrix `a` and the result vector is vector `b`. The returned vector will correspond to x. Source: Atilim Gunes Baydin, FsAlg, 2015, https://github.com/gbaydin/FsAlg
    let inline solve (a: ^T[,]) (b: ^T[]) =
        let lu, perm, _ = LUDecomposition a
        let bp = Array.init (a.GetLength(0)) (fun i -> b[perm[i]])
        matrixSolveHelper lu bp

    // Inverts matrix. Source: Atilim Gunes Baydin, FsAlg, 2015, https://github.com/gbaydin/FsAlg
    let inline inverseMatrix (m: ^T[,]) =
        let rows = m.GetLength(0)
        let res = Array2D.copy m
        let lu, perm, _ = LUDecomposition m
        let b:'T[] = Array.zeroCreate rows
        for i = 0 to rows - 1 do
            for j = 0 to rows - 1 do
                if i = perm[j] then
                    b[j] <- LanguagePrimitives.GenericOne<'T>
                else
                    b[j] <- LanguagePrimitives.GenericZero<'T>
            let x = matrixSolveHelper lu b
            res[0.., i] <- x
        res

    let inline InverseT(t: RawTensorCPU< ^T >) : RawTensorCPU< ^T > =
        Shape.checkCanInvert t.Shape
        let dim = t.Shape.Length
        if dim = 2 then  // One matrix
            let tinv = inverseMatrix (t.ToArray() :?> ^T[,])
            let tinvflat = [|  for i=0 to tinv.GetLength(0)-1 do for j=0 to tinv.GetLength(1)-1 do yield tinv[i, j] |]
            t.MakeLike(tinvflat, t.Shape) :?> RawTensorCPU<'T>
        else  // Batch of matrices
            let tinvs = 
                t.UnstackT(0)
                |> Array.map (fun v -> inverseMatrix (v.ToArray() :?> ^T[,]))
                |> Array.map (fun v -> [|  for i=0 to v.GetLength(0)-1 do for j=0 to v.GetLength(1)-1 do yield v[i, j] |])
                |> Array.map (fun v -> t.MakeLike(v, [|t.Shape[1]; t.Shape[2]|]))
            t.StackTs(tinvs, 0) :?> RawTensorCPU<'T>
    
    let inline diagonal(square: ^T[,]) =
        let n = square.GetLength(0)
        if n <> square.GetLength(1) then failwith "Expecting a square array"
        Array.init n (fun i -> square[i, i])

    let inline prod(t: ^T[]) =
        Array.fold (fun s x -> s * x) LanguagePrimitives.GenericOne<'T> t

    let inline DetT(t: RawTensorCPU< ^T >) : RawTensorCPU< ^T > =
        Shape.checkCanDet t.Shape
        let dim = t.Shape.Length
        if dim = 2 then
            let lu, _, toggle = LUDecomposition(t.ToArray() :?> ^T[,])
            let d:^T = toggle * (prod (diagonal lu))
            t.MakeLike([|d|], [||]) :?> RawTensorCPU<'T>
        else
            let tdets = 
                t.UnstackT(0)
                |> Array.map (fun v -> let lu, _, toggle = LUDecomposition(v.ToArray() :?> ^T[,]) in lu, toggle)
                |> Array.map (fun (lu, toggle) -> toggle * (prod (diagonal lu)))
                |> Array.map (fun v -> t.MakeLike([|v|], [||]))
            t.StackTs(tdets, 0) :?> RawTensorCPU<'T>

    let inline SolveTT(a: RawTensorCPU< ^T >, b: RawTensor) : RawTensorCPU< ^T > =
        let newShape = Shape.checkCanSolve a.Shape b.Shape
        let dimA = a.Shape.Length
        let dimB = b.Shape.Length
        if dimA = 2 then
            let n = a.Shape[0]
            let amatrix = (a.ToArray() :?> ^T[,])
            if dimB = 1 then
                let bvector = (b.ToArray() :?> ^T[])
                let s = solve amatrix bvector
                a.MakeLike(s, newShape) :?> RawTensorCPU<'T>
            else // dimB = 2
                let cols =
                    b.UnstackT(1) 
                    |> Array.map (fun v -> v.ToArray() :?> ^T[])
                    |> Array.map (fun v -> solve amatrix v)
                    |> Array.map (fun v -> a.MakeLike(v, [|n|]))
                a.StackTs(cols, 1) :?> RawTensorCPU<'T>
        else // dimA = 3
            let n = a.Shape[1]
            if dimB = 2 then
                let aa = a.UnstackT(0)
                let bb = b.UnstackT(0)
                let ss = 
                    Array.zip aa bb
                    |> Array.map (fun (aaa, bbb) ->
                                            let amatrix = (aaa.ToArray() :?> ^T[,])
                                            let bvector = (bbb.ToArray() :?> ^T[])
                                            let s = solve amatrix bvector
                                            a.MakeLike(s, [|n|]))
                a.StackTs(ss, 0) :?> RawTensorCPU<'T>
            else // dimB = 3
                let aa = a.UnstackT(0)
                let bb = b.UnstackT(0)
                let ss = 
                    Array.zip aa bb
                    |> Array.map (fun (aaa, bbb) ->
                                            let amatrix = (aaa.ToArray() :?> ^T[,])
                                            let cols =
                                                bbb.UnstackT(1)
                                                |> Array.map (fun v -> v.ToArray() :?> ^T[])
                                                |> Array.map (fun v -> solve amatrix v)
                                                |> Array.map (fun v -> a.MakeLike(v, [|n|]))
                                            a.StackTs(cols, 1))
                a.StackTs(ss, 0) :?> RawTensorCPU<'T>
            // failwithf "Unsupported shapes %A %A" a.Shape b.Shape

    let inline MaxPool1D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let batchSize, channels, inputSize, outputSize, outputShape =
            Shape.checkCanMaxpool1d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1[t1.MinIndexT()] - one
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v=0 to outputSize-1 do
                    let mutable maxvalue = minValue
                    let mutable maxindex = -1
                    for u=0 to kernelSize-1 do
                        let i = (v*stride) + u - padding
                        if i >= 0 && i < inputSize then
                            let value = t1[n, c, i]
                            if value > maxvalue then
                                maxvalue <- value
                                maxindex <- i
                    result[[|n; c; v|]] <- maxvalue
                    indices[[|n; c; v|]] <- maxindex
        result, indices

    let inline MaxPool2D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), outputShape =
            Shape.checkCanMaxpool2d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1[t1.MinIndexT()] - one
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable maxvalue = minValue
                        let mutable maxindexi0 = -1
                        let mutable maxindexi1 = -1
                        for u0=0 to kernelHeight-1 do
                            for u1=0 to kernelWidth-1 do
                                let i0 = (v0*stride[0]) + u0 - padding[0]
                                let i1 = (v1*stride[1]) + u1 - padding[1]
                                if i0 >= 0 && i0 < inputHeight && i1 >= 0 && i1 < inputWidth then
                                    let value = t1[n, c, i0, i1]
                                    if value > maxvalue then
                                        maxvalue <- value
                                        maxindexi0 <- i0
                                        maxindexi1 <- i1
                        result[[|n; c; v0; v1|]] <- maxvalue
                        indices[[|n; c; v0; v1|]] <- indexToFlatIndex [|inputHeight; inputWidth|] [|maxindexi0; maxindexi1|]
        result, indices

    let inline MaxPool3D(t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > * RawTensorCPU< int > =
        let (batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), outputShape) =
            Shape.checkCanMaxpool3d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let indices = t1.ZerosLike(outputShape, dtype=Int32) :?> RawTensorCPU<int>
        let minValue = t1[t1.MinIndexT()] - one
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
                                        let i0 = (v0*stride[0]) + u0 - padding[0]
                                        let i1 = (v1*stride[1]) + u1 - padding[1]
                                        let i2 = (v2*stride[2]) + u2 - padding[2]
                                        if i0 >= 0 && i0 < inputDepth && i1 >= 0 && i1 < inputHeight && i2 >= 0 && i2 < inputWidth then
                                            let value = t1[n, c, i0, i1, i2]
                                            if value > maxvalue then
                                                maxvalue <- value
                                                maxindexi0 <- i0
                                                maxindexi1 <- i1
                                                maxindexi2 <- i2
                            result[[|n; c; v0; v1; v2|]] <- maxvalue
                            indices[[|n; c; v0; v1; v2|]] <- indexToFlatIndex [|inputDepth; inputHeight; inputWidth|] [|maxindexi0; maxindexi1; maxindexi2|]
        result, indices

    let inline MaxUnpool1D(t1: RawTensorCPU< ^T >, indices: RawTensorCPU<int>, outputSize: int[]) : RawTensorCPU< ^T > =
        let batchSize, channels, inputSize, outputShape =
            Shape.checkCanMaxunpool1d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for u=0 to inputSize-1 do
                    let i = indices[[|n; c; u|]]
                    result[[|n; c; i|]] <- t1[[|n; c; u|]]
        result

    let inline MaxUnpool2D(t1: RawTensorCPU< ^T >, indices: RawTensorCPU<int>, outputSize:int[]) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputHeight, inputWidth), outputShape =
            Shape.checkCanMaxunpool2d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for u0=0 to inputHeight-1 do
                    for u1=0 to inputWidth-1 do
                        let iflat = indices[[|n; c; u0; u1|]]
                        let i = flatIndexToIndex [|outputSize[2]; outputSize[3]|] iflat
                        result[[|n; c; i[0]; i[1]|]] <- t1[[|n; c; u0; u1|]]
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
                            let iflat = indices[[|n; c; u0; u1; u2|]]
                            let i = flatIndexToIndex [|outputSize[2]; outputSize[3]; outputSize[4]|] iflat
                            result[[|n; c; i[0]; i[1]; i[2]|]] <- t1[[|n; c; u0; u1; u2|]]
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
                tshape[2] <- t1.Shape[2] + padding * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorCPU< ^T >
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v=0 to outputSize-1 do
                    let mutable value = zero
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelSize-1 do
                            value <- value + t2[k, c, u] * t1[n, c, (v*stride) + u]
                    result[[|n; k; v|]] <- value
        result

    let inline Conv2D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
        // t1: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
        // t2: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
        let batchSize, inputChannels, (kernelHeight, kernelWidth), (outputChannels, outputHeight, outputWidth), outputShape =
            Shape.checkCanConv2d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1|]
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU< ^T>
        let t1 =
            if padding[0] = 0 && padding[1] = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape[2] <- t1.Shape[2] + padding[0] * 2
                tshape[3] <- t1.Shape[3] + padding[1] * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding[0]; padding[1]|], t1) :?> RawTensorCPU< ^T >
        let t2 = t2 :?> RawTensorCPU< ^T >
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable value = zero
                        for c=0 to inputChannels-1 do
                            for u0=0 to kernelHeight-1 do
                                for u1=0 to kernelWidth-1 do
                                    value <- value + t2[k, c, u0, u1] * t1[n, c, (v0*stride[0])+u0, (v1*stride[1])+u1]
                        result[[|n; k; v0; v1|]] <- value
        result

    let inline Conv3D(t1: RawTensorCPU< ^T >, t2: RawTensor, stride: int[], padding: int[]) : RawTensorCPU< ^T > =
        // t1: input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth)
        // t2: filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth)
        let batchSize, inputChannels, (kernelDepth, kernelHeight, kernelWidth), (outputChannels, outputDepth, outputHeight, outputWidth), outputShape = 
            Shape.checkCanConv3d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1;1|]  
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU< ^T>
        let t1 =
            if padding[0] = 0 && padding[1] = 0 && padding[2] = 0 then
                t1
            else
                let tshape = Array.copy t1.Shape
                tshape[2] <- t1.Shape[2] + padding[0] * 2
                tshape[3] <- t1.Shape[3] + padding[1] * 2
                tshape[4] <- t1.Shape[4] + padding[2] * 2
                let t = t1.ZerosLike(tshape)
                t.AddTTSlice([|0; 0; padding[0]; padding[1]; padding[2]|], t1) :?> RawTensorCPU< ^T >
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
                                            value <- value + t2[k, c, u0, u1, u2] * t1[n, c, (v0*stride[0])+u0, (v1*stride[1])+u1, (v2*stride[2])+u2]
                            result[[|n; k; v0; v1; v2|]] <- value
        result

    let inline AvgPool1D ofInt (t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T >=
        let batchSize, channels, inputSize, outputSize, outputShape =
            Shape.checkCanAvgpool1d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v=0 to outputSize-1 do
                    let mutable avg = zero
                    for u=0 to kernelSize-1 do
                        let i = (v*stride) + u - padding
                        if i >= 0 && i < inputSize then
                            let value = t1[n, c, i]
                            avg <- avg + value
                    result[[|n; c; v|]] <- avg / ofInt kernelSize
        result

    let inline AvgPool2D ofInt (t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), outputShape =
            Shape.checkCanAvgpool2d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let kernelSize = kernelHeight * kernelWidth
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable avg = zero
                        for u0=0 to kernelHeight-1 do
                            for u1=0 to kernelWidth-1 do
                                let i0 = (v0*stride[0]) + u0 - padding[0]
                                let i1 = (v1*stride[1]) + u1 - padding[1]
                                if i0 >= 0 && i0 < inputHeight && i1 >= 0 && i1 < inputWidth then
                                    let value = t1[n, c, i0, i1]
                                    avg <- avg + value
                        result[[|n; c; v0; v1|]] <- avg / ofInt kernelSize
        result

    let inline AvgPool3D ofInt (t1: RawTensorCPU< ^T >, kernelSize, stride, padding) : RawTensorCPU< ^T > =
        let (batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), outputShape) =
            Shape.checkCanAvgpool3d t1.Dtype t1.Shape kernelSize stride padding
        let result = t1.ZerosLike(outputShape) :?> RawTensorCPU<'T>
        let kernelSize = kernelDepth * kernelHeight * kernelWidth
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputDepth-1 do
                    for v1=0 to outputHeight-1 do
                        for v2=0 to outputWidth-1 do
                            let mutable avg = zero
                            for u0=0 to kernelDepth-1 do
                                for u1=0 to kernelHeight-1 do
                                    for u2=0 to kernelWidth-1 do
                                        let i0 = (v0*stride[0]) + u0 - padding[0]
                                        let i1 = (v1*stride[1]) + u1 - padding[1]
                                        let i2 = (v2*stride[2]) + u2 - padding[2]
                                        if i0 >= 0 && i0 < inputDepth && i1 >= 0 && i1 < inputHeight && i2 >= 0 && i2 < inputWidth then
                                            let value = t1[n, c, i0, i1, i2]
                                            avg <- avg + value
                            result[[|n; c; v0; v1; v2|]] <- avg / ofInt kernelSize
        result

    let inline AvgPoolReverse1D ofInt (t1: RawTensorCPU< ^T >, originalInput: RawTensor, kernelSize, stride, padding) : RawTensorCPU< ^T > =
        let batchSize, channels, inputSize, outputSize, _outputShape =
            Shape.checkCanAvgpool1d t1.Dtype originalInput.Shape kernelSize stride padding
        let result = t1.ZerosLike(originalInput.Shape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v=0 to outputSize-1 do
                    for u=0 to kernelSize-1 do
                        let i = (v*stride) + u - padding
                        if i >= 0 && i < inputSize then
                            result[[|n; c; i|]] <- t1[[|n; c; v|]] / ofInt kernelSize
        result

    let inline AvgPoolReverse2D ofInt (t1: RawTensorCPU< ^T >, originalInput: RawTensor, kernelSize, stride, padding) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), _outputShape =
            Shape.checkCanAvgpool2d t1.Dtype originalInput.Shape kernelSize stride padding
        let kernelSize = kernelHeight * kernelWidth
        let result = t1.ZerosLike(originalInput.Shape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        for u0=0 to kernelHeight-1 do
                            for u1=0 to kernelWidth-1 do
                                let i0 = (v0*stride[0]) + u0 - padding[0]
                                let i1 = (v1*stride[1]) + u1 - padding[1]
                                if i0 >= 0 && i0 < inputHeight && i1 >= 0 && i1 < inputWidth then
                                    result[[|n; c; i0; i1|]] <- t1[[|n; c; v0; v1|]] / ofInt kernelSize
        result

    let inline AvgPoolReverse3D ofInt (t1: RawTensorCPU< ^T >, originalInput: RawTensor, kernelSize, stride, padding) : RawTensorCPU< ^T > =
        let batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), _outputShape =
            Shape.checkCanAvgpool3d t1.Dtype originalInput.Shape kernelSize stride padding
        let kernelSize = kernelDepth * kernelHeight * kernelWidth
        let result = t1.ZerosLike(originalInput.Shape) :?> RawTensorCPU<'T>
        for n=0 to batchSize-1 do
            for c=0 to channels-1 do
                for v0=0 to outputDepth-1 do
                    for v1=0 to outputHeight-1 do
                        for v2=0 to outputWidth-1 do
                            for u0=0 to kernelDepth-1 do
                                for u1=0 to kernelHeight-1 do
                                    for u2=0 to kernelWidth-1 do
                                        let i0 = (v0*stride[0]) + u0 - padding[0]
                                        let i1 = (v1*stride[1]) + u1 - padding[1]
                                        let i2 = (v2*stride[2]) + u2 - padding[2]
                                        if i0 >= 0 && i0 < inputDepth && i1 >= 0 && i1 < inputHeight && i2 >= 0 && i2 < inputWidth then
                                            result[[|n; c; i0; i1; i2|]] <- t1[[|n; c; v0; v1; v2|]] / ofInt kernelSize
        result

    let inline NegT op (t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = Array.map op t.Values
        (result, t.Shape)

    let inline SumT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        if Array.isEmpty t.Values then ([|zero< ^T >|], Shape.scalar) else // Return a zero-valued scalar tensor if summing a zero-sized tensor (not holding any value). This is mirroring the behavior in PyTorch 1.5.1.
        let result = Array.reduce (+) t.Values
        ([|result|], [||])
    
    let inline SumTDim(t: RawTensorCPU< ^T >, dim: int) : RawTensorCPU< ^T > =
        let sBounds = Array2D.init t.Dim 3 (fun i j -> if j=0 then 0 elif j=1 then t.Shape[i]-1 else 0)
        sBounds[dim, 1] <- 0
        sBounds[dim, 2] <- 1
        let s = t.ZerosLike(shape=t.Shape, dtype=t.Dtype.SummationType).GetSlice(sBounds) :?> RawTensorCPU<'T>
        s.SetMutable()
        for i=0 to t.Shape[dim]-1 do
            sBounds[dim,0] <- i
            sBounds[dim,1] <- i
            sBounds[dim,2] <- 1
            s.AddInPlace(t.GetSlice(sBounds).Cast(t.Dtype.SummationType))
        s

    let inline SignT op (t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map op
        (result, t.Shape)

    let inline FloorT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map floor
        (result, t.Shape)

    let inline CeilT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map ceil
        (result, t.Shape)

    let inline RoundT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map round
        (result, t.Shape)

    let inline AbsT op (t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map op
        (result, t.Shape)

    let inline ReluT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map (max zero< ^T >) 
        (result, t.Shape)

    let inline SoftplusT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map (fun x -> (max zero< ^T > x) + log(one< ^T > + exp(-abs(x))))
        (result, t.Shape)

    let inline SigmoidT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map (fun v -> one / (one + exp -v))
        (result, t.Shape)

    let inline ExpT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map exp
        (result, t.Shape)

    let inline LogT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map log
        (result, t.Shape)

    let inline Log10T(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map log10
        (result, t.Shape)
        
    let inline SqrtT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map sqrt
        (result, t.Shape)
        
    let inline SinT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map sin
        (result, t.Shape)
        
    let inline CosT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map cos
        (result, t.Shape)                
        
    let inline TanT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map tan
        (result, t.Shape)
        
    let inline SinhT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map sinh
        (result, t.Shape)
        
    let inline CoshT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map cosh
        (result, t.Shape)                
        
    let inline TanhT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map tanh
        (result, t.Shape)

    let inline AsinT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map asin
        (result, t.Shape)
        
    let inline AcosT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map acos
        (result, t.Shape)                
        
    let inline AtanT(t: RawTensorCPU< ^T >) : (^T[] * Shape) =
        let result = t.Values |> Array.map atan
        (result, t.Shape)

    let inline Random ofDouble (shape:Shape) : (^T[] * Shape) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Uniform()))
        (values, shape)

    let inline RandomNormal ofDouble (shape:Shape) : (^T[] * Shape) =
        let values = Array.init (shapeLength shape) (fun _ -> ofDouble (DiffSharp.Util.Random.Normal()))
        (values, shape)

    let inline RandomInt ofInt (shape:Shape) (low:int) (high:int) : (^T[] * Shape) =
        let values = Array.init (shapeLength shape) (fun _ -> ofInt (DiffSharp.Util.Random.Integer(low, high)))
        (values, shape)

/// The concrete implementation of RawTensor for Float32 data.
type RawTensorFloat32(values: float32[], shape:Shape, device) =
    inherit RawTensorCPU<float32>(values, shape, Dtype.Float32, device)
    let create(values, shape) : RawTensor = upcast RawTensorFloat32(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device) 
    static let createOn device (values, shape) : RawTensor = upcast RawTensorFloat32(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toSingle(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toSingle(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toSingle()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toSingle()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toSingle(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toSingle()) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t2.PowFromT0T(t1) = RawTensorCPU.PowT0T(t1.toSingle(), t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2.toSingle()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumTDim(dim, resultType) =
        let res = RawTensorCPU.SumTDim(t, dim)
        match resultType with 
        | None -> res :> _
        | Some dtype -> res.Cast(dtype)
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
    override t.InverseT() = RawTensorCPU.InverseT(t) :> _
    override t.DetT() = RawTensorCPU.DetT(t) :> _
    override a.SolveTT(b) = RawTensorCPU.SolveTT(a, b) :> _

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toSingle()) |> createOn device
    static member Random(shape:Shape, device) = RawTensorCPU.Random float32 shape |> createOn device
    static member RandomNormal(shape:Shape, device) = RawTensorCPU.RandomNormal float32 shape |> createOn device
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt float32 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorFloat64(values: double[], shape:Shape, device) =
    inherit RawTensorCPU<double>(values, shape, Dtype.Float64, device)

    let create(values, shape) : RawTensor = upcast RawTensorFloat64(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorFloat64(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toDouble() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toDouble() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toDouble(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toDouble(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toDouble()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toDouble()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toDouble(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toDouble()) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t2.PowFromT0T(t1) = RawTensorCPU.PowT0T(t1.toDouble(), t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2.toDouble()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D double (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D double (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D double (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D double (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D double (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D double (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumTDim(dim, resultType) =
        let res = RawTensorCPU.SumTDim(t, dim)
        match resultType with 
        | None -> res :> _
        | Some dtype -> res.Cast(dtype)
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
    override t.InverseT() = RawTensorCPU.InverseT(t) :> _
    override t.DetT() = RawTensorCPU.DetT(t) :> _
    override a.SolveTT(b) = RawTensorCPU.SolveTT(a, b) :> _

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toDouble()) |> createOn device
    static member Random(shape:Shape, device) = RawTensorCPU.Random double shape |> createOn device
    static member RandomNormal(shape:Shape, device) = RawTensorCPU.RandomNormal double shape |> createOn device
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt double shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorInt8(values: int8[], shape:Shape, device) =
    inherit RawTensorCPU<int8>(values, shape, Dtype.Int8, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt8(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorInt8(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSByte() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSByte() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toSByte(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toSByte(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toSByte()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toSByte()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toSByte(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toSByte()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D int8 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D int8 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D int8 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D int8 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D int8 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D int8 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumTDim(dim, resultType) = t.Cast(Dtype.Int64).SumTDim(dim, ?resultType=resultType)
    override t.SignT() = RawTensorCPU.SignT (sign >> int8) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toSByte()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Int8
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Int8
    static member RandomInt(shape, low, high, device) = RawTensorCPU.RandomInt int8 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorByte(values: byte[], shape:Shape, device) =
    inherit RawTensorCPU<byte>(values, shape, Dtype.Byte, device)

    let create(values, shape) : RawTensor = upcast RawTensorByte(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorByte(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toByte() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toByte() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toByte(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toByte(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toByte()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toByte()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toByte(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toByte()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D byte (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D byte (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D byte (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D byte (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D byte (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D byte (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (sbyte >> (~-) >> byte ) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumTDim(dim, resultType) = t.Cast(Dtype.Int64).SumTDim(dim, ?resultType=resultType)
    override t.SignT() = RawTensorCPU.SignT (min 1uy) t |> create
    override t.AbsT() = RawTensorCPU.AbsT id t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toByte()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Byte
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Byte
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt byte shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorInt16(values: int16[], shape:Shape, device) =
    inherit RawTensorCPU<int16>(values, shape, Dtype.Int16, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt16(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorInt16(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt16() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt16() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toInt16(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toInt16(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toInt16()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toInt16()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toInt16(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toInt16()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D int16 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D int16 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D int16 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D int16 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D int16 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D int16 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumTDim(dim, resultType) = t.Cast(Dtype.Int64).SumTDim(dim, ?resultType=resultType)
    override t.SignT() = RawTensorCPU.SignT (sign >> int16) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toInt16()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Int16
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Int16
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt int16 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorInt32(values: int32[], shape:Shape, device) =
    inherit RawTensorCPU<int32>(values, shape, Dtype.Int32, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt32(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorInt32(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt32() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt32() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toInt32(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toInt32(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toInt32()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toInt32()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toInt32(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toInt32()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D int32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D int32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D int32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D int32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D int32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D int32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) = t.Cast(Dtype.Int64).SumT(?resultType=resultType)
    override t.SumTDim(dim, resultType) = t.Cast(Dtype.Int64).SumTDim(dim, ?resultType=resultType)
    override t.SignT() = RawTensorCPU.SignT (sign >> int32) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toInt32()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Int32
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Int32
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt int32 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorInt64(values: int64[], shape:Shape, device) =
    inherit RawTensorCPU<int64>(values, shape, Dtype.Int64, device)

    let create(values, shape) : RawTensor = upcast RawTensorInt64(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorInt64(values, shape, device)

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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt64() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toInt64() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toInt64(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toInt64(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toInt64()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toInt64()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toInt64(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toInt64()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D int64 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D int64 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D int64 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D int64 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D int64 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D int64 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumTDim(dim, resultType) =
        let res = RawTensorCPU.SumTDim(t, dim)
        match resultType with 
        | None -> res :> _
        | Some dtype -> res.Cast(dtype)
    override t.SignT() = RawTensorCPU.SignT (sign >> int64) t |> create
    override t.AbsT() = RawTensorCPU.AbsT abs t |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toInt64()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Int64
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Int64
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt int64 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

type RawTensorBool(values: bool[], shape:Shape, device) =
    inherit RawTensorCPU<bool>(values, shape, Dtype.Bool, device)

    let create(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)
    static let createOn device (values, shape) : RawTensor = upcast RawTensorBool(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorBool(values, shape, defaultArg newDevice device)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
    override t1.LtTT(t2) = t1.MakeLike(Array.map2 (<) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.GtTT(t2) = t1.MakeLike(Array.map2 (>) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.LeTT(t2) = t1.MakeLike(Array.map2 (<=) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.GeTT(t2) = t1.MakeLike(Array.map2 (>=) t1.Values (t2.GetTypedValues()), t1.Shape) 
    override t1.EqTT(t2) = RawTensorCPU.EqTT(t1, t2) |> create
    override t1.NeqTT(t2) = RawTensorCPU.NeqTT(t1, t2) |> create
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) = 
        let alpha = match alpha with Some v -> v.toBool() | None -> true
        t1.MakeLike(Array.map2 (||) t1.Values (Array.map (fun x -> alpha && x) (t2.GetTypedValues())), t1.Shape)
    override t1.AddTT0(t2, alpha) =
        let t2 = t2.toBool() 
        let alpha = match alpha with Some v -> v.toBool() | None -> true
        let values = Array.map (fun a -> a || (alpha && t2)) t1.Values
        t1.MakeLike(values, t1.Shape)
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((||), t1, location, t2) |> create
    override t1.MulTT(t2) = t1.MakeLike(Array.map2 (&&) t1.Values (t2.GetTypedValues()), t1.Shape)
    override t1.MulTT0(t2) = 
        let t2 = t2.toBool() 
        t1.MakeLike(Array.map (fun a -> a && t2) t1.Values, t1.Shape)
    override t.SumT(resultType) = t.Cast(Int64).SumT(?resultType=resultType)
    override t.SumTDim(dim, resultType) = t.Cast(Dtype.Int64).SumTDim(dim, ?resultType=resultType)
    override t.SignT() = t :> _

    override t.ClampT(_low, _high) = opNotSupported "Clamp" t.Dtype
    override t1.SubTT(t2) = opNotSupported2 "SubTT" t1.Dtype t2.Dtype
    override t2.SubFromT0T(_t1) = opNotSupported "SubT0T" t2.Dtype
    override t1.SubTT0(_t2) = opNotSupported "SubTT0" t1.Dtype
    override t1.DivTT(t2) = opNotSupported2 "DivTT" t1.Dtype t2.Dtype
    override t2.DivFromT0T(_t1) = opNotSupported "DivT0T" t2.Dtype
    override t1.DivTT0(_t2) = opNotSupported "DivTT0" t1.Dtype
    override t1.MatMulTT(t2) = opNotSupported2 "MatMulTT" t1.Dtype t2.Dtype
    override t1.BMMTT(t2) = opNotSupported2 "BMMTT" t1.Dtype t2.Dtype
    override t1.MaxPool1D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool1D" t1.Dtype
    override t1.MaxPool2D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool2D" t1.Dtype
    override t1.MaxPool3D(_kernelSize, _stride, _padding) = opNotSupported "MaxPool3D" t1.Dtype
    override t1.MaxUnpool1D(_indices, _outputSize) = opNotSupported "MaxUnpool1D" t1.Dtype
    override t1.MaxUnpool2D(_indices, _outputSize) = opNotSupported "MaxUnpool2D" t1.Dtype
    override t1.MaxUnpool3D(_indices, _outputSize) = opNotSupported "MaxUnpool3D" t1.Dtype
    override t1.Conv1D(t2, _stride, _padding) = opNotSupported2 "Conv1D" t1.Dtype t2.Dtype
    override t1.Conv2D(t2, _stride, _padding) = opNotSupported2 "Conv2D" t1.Dtype t2.Dtype
    override t1.Conv3D(t2, _stride, _padding) = opNotSupported2 "Conv3D" t1.Dtype t2.Dtype
    override t1.AvgPool1D(_kernelSize, _stride, _padding) = opNotSupported "AvgPool1D" t1.Dtype
    override t1.AvgPool2D(_kernelSize, _stride, _padding) = opNotSupported "AvgPool2D" t1.Dtype
    override t1.AvgPool3D(_kernelSize, _stride, _padding) = opNotSupported "AvgPool3D" t1.Dtype
    override t1.AvgPoolReverse1D(_originalInput, _kernelSize, _stride, _padding) = opNotSupported "AvgPoolReverse1D" t1.Dtype
    override t1.AvgPoolReverse2D(_originalInput, _kernelSize, _stride, _padding) = opNotSupported "AvgPoolReverse2D" t1.Dtype
    override t1.AvgPoolReverse3D(_originalInput, _kernelSize, _stride, _padding) = opNotSupported "AvgPoolReverse3D" t1.Dtype
    override t.NegT() = opNotSupported "NegT" t.Dtype
    override t.AbsT() = opNotSupported "AbsT" t.Dtype
    override t.ReluT() = opNotSupported "ReluT" t.Dtype
    override t.SoftplusT() = opNotSupported "SoftplusT" t.Dtype
    override t1.PowTT(t2) = opNotSupported2 "PowTT" t1.Dtype t2.Dtype
    override t2.PowFromT0T(_t1) = opNotSupported "PowT0T" t2.Dtype
    override t1.PowTT0(_t2) = opNotSupported "PowTT0" t1.Dtype
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = ([| false |], Shape.scalar) |> createOn device
    static member One(device) = ([| true |], Shape.scalar) |> createOn device
    static member Zeros(shape:Shape, device) = (Array.zeroCreate (shapeLength shape), shape) |> createOn device
    static member Empty(shape:Shape, device) = (Array.zeroCreate (shapeLength shape), shape) |> createOn device
    static member Ones(shape:Shape, device) = (Array.create (shapeLength shape) true, shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toBool()) |> createOn device
    static member Random(_shape:Shape, _device) = opNotSupported "Random" Dtype.Bool
    static member RandomNormal(_shape:Shape, _device) = opNotSupported "RandomNormal" Dtype.Bool
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt System.Convert.ToBoolean shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

/// The concrete implementation of RawTensor for Float16 data.
type RawTensorFloat16(values: float32[], shape:Shape, device) =
    inherit RawTensorCPU<float32>(values, shape, Dtype.Float16, device)
    let create(values, shape) : RawTensor = upcast RawTensorFloat16(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device) 
    static let createOn device (values, shape) : RawTensor = upcast RawTensorFloat16(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorFloat16(values, shape, defaultArg newDevice device)
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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toSingle(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toSingle(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toSingle()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toSingle()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toSingle(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toSingle()) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t2.PowFromT0T(t1) = RawTensorCPU.PowT0T(t1.toSingle(), t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2.toSingle()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumTDim(dim, resultType) =
        let res = RawTensorCPU.SumTDim(t, dim)
        match resultType with 
        | None -> res :> _
        | Some dtype -> res.Cast(dtype)
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toSingle()) |> createOn device
    static member Random(shape:Shape, device) = RawTensorCPU.Random float32 shape |> createOn device
    static member RandomNormal(shape:Shape, device) = RawTensorCPU.RandomNormal float32 shape |> createOn device
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt float32 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

/// The concrete implementation of RawTensor for Float16 data.
type RawTensorBFloat16(values: float32[], shape:Shape, device) =
    inherit RawTensorCPU<float32>(values, shape, Dtype.BFloat16, device)
    let create(values, shape) : RawTensor = upcast RawTensorBFloat16(values, shape, device)
    let createBool(values, shape) : RawTensor = upcast RawTensorBool(values, shape, device) 
    static let createOn device (values, shape) : RawTensor = upcast RawTensorBFloat16(values, shape, device)

    override t.MakeLike(values, shape, newDevice) = upcast RawTensorBFloat16(values, shape, defaultArg newDevice device)
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
    override t.MaxReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (>) (t, dim, keepDim)
    override t.MinReduceT(dim, keepDim) = RawTensorCPU.MinMaxReduceT (<) (t, dim, keepDim)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT(t1, t2, alpha) |> create
    override t1.AddTT0(t2, alpha) =
        let alpha = match alpha with Some v -> v.toSingle() | None -> RawTensorCPU.one
        RawTensorCPU.AddTT0(t1, t2.toSingle(), alpha) |> create
    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
    override t2.SubFromT0T(t1) = RawTensorCPU.SubT0T(t1.toSingle(), t2) |> create
    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2.toSingle()) |> create
    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2.toSingle()) |> create
    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
    override t2.DivFromT0T(t1) = RawTensorCPU.DivT0T(t1.toSingle(), t2) |> create
    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2.toSingle()) |> create
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
    override t2.PowFromT0T(t1) = RawTensorCPU.PowT0T(t1.toSingle(), t2) |> create
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2.toSingle()) |> create
    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.BMMTT(t2) = RawTensorCPU.BMMTT(t1, t2) |> create
    override t1.MaxPool1D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool1D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool2D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool2D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxPool3D(kernelSize, stride, padding) = let result, indices = RawTensorCPU.MaxPool3D(t1, kernelSize, stride, padding) in result :> _, indices :> _
    override t1.MaxUnpool1D(indices, outputSize) = RawTensorCPU.MaxUnpool1D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool2D(indices, outputSize) = RawTensorCPU.MaxUnpool2D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.MaxUnpool3D(indices, outputSize) = RawTensorCPU.MaxUnpool3D(t1, indices :?> RawTensorCPU<int>, outputSize) :> _
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t1.AvgPool1D(kernelSize, stride, padding) = RawTensorCPU.AvgPool1D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool2D(kernelSize, stride, padding) = RawTensorCPU.AvgPool2D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPool3D(kernelSize, stride, padding) = RawTensorCPU.AvgPool3D float32 (t1, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse1D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse2D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) = RawTensorCPU.AvgPoolReverse3D float32 (t1, originalInput, kernelSize, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT (~-) (t) |> create
    override t.SumT(resultType) =
        let res = RawTensorCPU.SumT(t) |> create
        match resultType with 
        | None -> res
        | Some dtype -> res.Cast(dtype)
    override t.SumTDim(dim, resultType) =
        let res = RawTensorCPU.SumTDim(t, dim)
        match resultType with 
        | None -> res :> _
        | Some dtype -> res.Cast(dtype)
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
    override t.InverseT() = opNotSupported "InverseT" t.Dtype
    override t.DetT() = opNotSupported "DetT" t.Dtype
    override a.SolveTT(_) = opNotSupported "SolveTT" a.Dtype

    static member Seed(seed) = Random.Seed(seed)
    static member Zero(device) = RawTensorCPU.Zero() |> createOn device
    static member One(device) = RawTensorCPU.One() |> createOn device
    static member Zeros(shape:Shape, device) = RawTensorCPU.Zeros(shape) |> createOn device
    static member Empty(shape:Shape, device) = RawTensorCPU.Empty(shape) |> createOn device
    static member Ones(shape:Shape, device) = RawTensorCPU.Ones(shape) |> createOn device
    static member Full(shape:Shape, value:scalar, device) = RawTensorCPU.Full (shape, value.toSingle()) |> createOn device
    static member Random(shape:Shape, device) = RawTensorCPU.Random float32 shape |> createOn device
    static member RandomNormal(shape:Shape, device) = RawTensorCPU.RandomNormal float32 shape |> createOn device
    static member RandomInt(shape:Shape, low:int, high:int, device) = RawTensorCPU.RandomInt float32 shape low high |> createOn device
    static member CreateFromFlatArray(values:Array, shape, device) = RawTensorCPU.CreateFromFlatArray (values, shape) |> createOn device

#if TEST_DUPLICATE_BACKEND
type TestDuplicateBackendTensorStatics() = 
#else
type ReferenceBackendTensorStatics() = 
#endif

    inherit BackendTensorStatics()

    override _.GetDevices(deviceType) =
        match deviceType with 
        | None -> [ Device.CPU (* ; Device.GPU *) ]
        | Some DeviceType.CPU -> [ Device.CPU]
        //| Some DeviceType.CUDA -> [ Device.GPU ]
        | Some _ -> []

    override _.IsDeviceTypeAvailable (deviceType) = (match deviceType with DeviceType.CPU -> true | _ -> false)
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.Zero(device)
        | BFloat16 -> RawTensorBFloat16.Zero(device)
        | Float32 -> RawTensorFloat32.Zero(device)
        | Float64 -> RawTensorFloat64.Zero(device)
        | Int8 -> RawTensorInt8.Zero(device)
        | Byte -> RawTensorByte.Zero(device)
        | Int16 -> RawTensorInt16.Zero(device)
        | Int32 -> RawTensorInt32.Zero(device)
        | Int64 -> RawTensorInt64.Zero(device)
        | Bool -> RawTensorBool.Zero(device)
    override _.One(dtype, device) = 
        match dtype with 
        | Float16 -> RawTensorFloat16.One(device)
        | BFloat16 -> RawTensorBFloat16.One(device)
        | Float32 -> RawTensorFloat32.One(device)
        | Float64 -> RawTensorFloat64.One(device)
        | Int8 -> RawTensorInt8.One(device)
        | Byte -> RawTensorByte.One(device)
        | Int16 -> RawTensorInt16.One(device)
        | Int32 -> RawTensorInt32.One(device)
        | Int64 -> RawTensorInt64.One(device)
        | Bool -> RawTensorBool.One(device)
    override _.Zeros(shape:Shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.Zeros(shape, device)
        | BFloat16 -> RawTensorBFloat16.Zeros(shape, device)
        | Float32 -> RawTensorFloat32.Zeros(shape, device)
        | Float64 -> RawTensorFloat64.Zeros(shape, device)
        | Int8 -> RawTensorInt8.Zeros(shape, device)
        | Byte -> RawTensorByte.Zeros(shape, device)
        | Int16 -> RawTensorInt16.Zeros(shape, device)
        | Int32 -> RawTensorInt32.Zeros(shape, device)
        | Int64 -> RawTensorInt64.Zeros(shape, device)
        | Bool -> RawTensorBool.Zeros(shape, device)
    override _.Empty(shape:Shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.Empty(shape, device)
        | BFloat16 -> RawTensorBFloat16.Empty(shape, device)
        | Float32 -> RawTensorFloat32.Empty(shape, device)
        | Float64 -> RawTensorFloat64.Empty(shape, device)
        | Int8 -> RawTensorInt8.Empty(shape, device)
        | Byte -> RawTensorByte.Empty(shape, device)
        | Int16 -> RawTensorInt16.Empty(shape, device)
        | Int32 -> RawTensorInt32.Empty(shape, device)
        | Int64 -> RawTensorInt64.Empty(shape, device)
        | Bool -> RawTensorBool.Empty(shape, device)
    override _.Ones(shape:Shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.Ones(shape, device)
        | BFloat16 -> RawTensorBFloat16.Ones(shape, device)
        | Float32 -> RawTensorFloat32.Ones(shape, device)
        | Float64 -> RawTensorFloat64.Ones(shape, device)
        | Int8 -> RawTensorInt8.Ones(shape, device)
        | Byte -> RawTensorByte.Ones(shape, device)
        | Int16 -> RawTensorInt16.Ones(shape, device)
        | Int32 -> RawTensorInt32.Ones(shape, device)
        | Int64 -> RawTensorInt64.Ones(shape, device)
        | Bool -> RawTensorBool.Ones(shape, device)
    override _.Full(shape:Shape, value:scalar, dtype, device) = 
        match dtype with 
        | Float16 -> RawTensorFloat16.Full(shape, value, device)
        | BFloat16 -> RawTensorBFloat16.Full(shape, value, device)
        | Float32 -> RawTensorFloat32.Full(shape, value, device)
        | Float64 -> RawTensorFloat64.Full(shape, value, device)
        | Int8 -> RawTensorInt8.Full(shape, value, device)
        | Byte -> RawTensorByte.Full(shape, value, device)
        | Int16 -> RawTensorInt16.Full(shape, value, device)
        | Int32 -> RawTensorInt32.Full(shape, value, device)
        | Int64 -> RawTensorInt64.Full(shape, value, device)
        | Bool -> RawTensorBool.Full(shape, value, device)
    override _.Random(shape:Shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.Random(shape, device)
        | BFloat16 -> RawTensorBFloat16.Random(shape, device)
        | Float32 -> RawTensorFloat32.Random(shape, device)
        | Float64 -> RawTensorFloat64.Random(shape, device)
        | Int8 -> RawTensorInt8.Random(shape, device)
        | Byte -> RawTensorByte.Random(shape, device)
        | Int16 -> RawTensorInt16.Random(shape, device)
        | Int32 -> RawTensorInt32.Random(shape, device)
        | Int64 -> RawTensorInt64.Random(shape, device)
        | Bool -> RawTensorBool.Random(shape, device)
    override _.RandomNormal(shape:Shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.RandomNormal(shape, device)
        | BFloat16 -> RawTensorBFloat16.RandomNormal(shape, device)
        | Float32 -> RawTensorFloat32.RandomNormal(shape, device)
        | Float64 -> RawTensorFloat64.RandomNormal(shape, device)
        | Int8 -> RawTensorInt8.RandomNormal(shape, device)
        | Byte -> RawTensorByte.RandomNormal(shape, device)
        | Int16 -> RawTensorInt16.RandomNormal(shape, device)
        | Int32 -> RawTensorInt32.RandomNormal(shape, device)
        | Int64 -> RawTensorInt64.RandomNormal(shape, device)
        | Bool -> RawTensorBool.RandomNormal(shape, device)
    override _.RandomInt(shape:Shape, low:int, high:int, dtype, device) = 
        match dtype with 
        | Float16 -> RawTensorFloat16.RandomInt(shape, low, high, device)
        | BFloat16 -> RawTensorBFloat16.RandomInt(shape, low, high, device)
        | Float32 -> RawTensorFloat32.RandomInt(shape, low, high, device)
        | Float64 -> RawTensorFloat64.RandomInt(shape, low, high, device)
        | Int8 -> RawTensorInt8.RandomInt(shape, low, high, device)
        | Byte -> RawTensorByte.RandomInt(shape, low, high, device)
        | Int16 -> RawTensorInt16.RandomInt(shape, low, high, device)
        | Int32 -> RawTensorInt32.RandomInt(shape, low, high, device)
        | Int64 -> RawTensorInt64.RandomInt(shape, low, high, device)
        | Bool -> RawTensorBool.RandomInt(shape, low, high, device)
    override _.CreateFromFlatArray(values:Array, shape, dtype, device) =
        match dtype with 
        | Float16 -> RawTensorFloat16.CreateFromFlatArray(values, shape, device)
        | BFloat16 -> RawTensorBFloat16.CreateFromFlatArray(values, shape, device)
        | Float32 -> RawTensorFloat32.CreateFromFlatArray(values, shape, device)
        | Float64 -> RawTensorFloat64.CreateFromFlatArray(values, shape, device)
        | Int8 -> RawTensorInt8.CreateFromFlatArray(values, shape, device)
        | Byte -> RawTensorByte.CreateFromFlatArray(values, shape, device)
        | Int16 -> RawTensorInt16.CreateFromFlatArray(values, shape, device)
        | Int32 -> RawTensorInt32.CreateFromFlatArray(values, shape, device)
        | Int64 -> RawTensorInt64.CreateFromFlatArray(values, shape, device)
        | Bool -> RawTensorBool.CreateFromFlatArray(values, shape, device)

