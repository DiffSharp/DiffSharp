namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
[<AbstractClass>]
type RawTensorCPU<'T>(values: 'T[], shape: int[], dtype: DType) =
    inherit RawTensor(shape, dtype, CPU, DiffSharp.Backend.Backend.None)

    member __.Values = values

    member internal t.IndexToFlatIndex(index:int[]) =
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) t.Shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        flatIndex
    
    member internal t.FlatIndexToIndex(flatIndex:int) =
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
            t.Values.[t.IndexToFlatIndex(index)]
        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then invalidArg "index" (sprintf "Expecting a %id index" t.Dim)
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(index:int[]) = t.Create(t.[index])
    
    override t.GetSlice(bounds:int[,]) =
        // if bounds.GetLength(0) <> t.Dim then invalidArg "bounds" (sprintf "Expecting %i-by-2 bounds" t.Dim)
        // printfn "%A" bounds
        let shape = Array.init (bounds.GetLength(0)) (fun i -> bounds.[i,1] - bounds.[i,0] + 1) |> shapeSqueeze -1
        // printfn "%A" shape
        let array = Array.zeroCreate (shapeLength shape)
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
        t.CreateShaped(array, shape)

    override t.Copy() = t.CreateShaped(Array.copy t.Values, Array.copy t.Shape)

    override t.CreateFromScalar(value: obj, shape) =
        let value = value:?>'T
        match shape.Length with
        | 0 -> t.CreateShaped([|value|], [||])
        | _ -> t.CreateShaped(Array.create (shapeLength shape) value, shape)

    abstract member CreateShaped: values: 'T[] * shape: int[] -> RawTensor

    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        match t.Dim with
        | 0 -> sprintf "%A" t.Values.[0]
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
        | 0 -> upcast t.Values.[0]
        | _ -> invalidOp (sprintf "Cannot convert %Ad Tensor to scalar" t.Dim)

    override t.ToArray() =
        match t.Dim with
        | 0 -> invalidOp "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> invalidOp (sprintf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape)

    override t.StackTs(tensors) =
        let tensors = tensors |> Seq.toList
        let values, shapes = tensors |> List.map (fun tensor -> (tensor :?> RawTensorCPU<'T>).Values, tensor.Shape) |> List.unzip
        if not (allEqual shapes) then invalidArg "tensors" "Expecting Tensors with same shape"
        let n = tensors |> List.length
        let m = shapeLength shapes.[0]
        let result = Array.zeroCreate (n * m)
        for i=0 to n-1 do
            for j=0 to m-1 do
                result.[i*m+j] <- values.[i].[j]
        t.CreateShaped(result, Array.append [|n|] shapes.[0])

    override t.UnstackT() =
        if t.Dim < 1 then invalidOp "Cannot unstack scalar Tensor (dim < 1)"
        let n = t.Shape.[0]
        let unstackedShape = if t.Dim = 1 then [||] else t.Shape |> Array.skip 1
        let unstackedLength = shapeLength unstackedShape
        Seq.init n (fun i -> Array.init unstackedLength (fun j -> t.Values.[i*unstackedLength+j]))
        |> Seq.map (fun v -> t.CreateShaped(v, unstackedShape))

    override t.TransposeT2() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
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
        if dims.Length > t.Dim then invalidOp <| sprintf "Expecting dims (list of dimension indices to flip) of length less than the Tensor's dimensions, received %A, %A" dims.Length t.Dim
        if hasDuplicates dims then invalidOp <| sprintf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
        if (Array.max dims) >= t.Dim then invalidOp <| sprintf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims t.Dim
        match t.Dim with
        | 0 -> t.Copy()
        | _ ->
            let result = t.Zeros(t.Shape) :?> RawTensorCPU<'T>
            let rec flip (shape:int[]) externalCoords = 
                let currentDim = t.Shape.Length - shape.Length
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[mirrorCoordinates globalCoords t.Shape dims] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        flip shape.[1..] (Array.append externalCoords [|i|])
            flip t.Shape [||]        
            upcast result

    override t.ViewT(shape:int[]) =
        if shapeLength t.Shape <> shapeLength shape then invalidOp <| sprintf "Cannot view Tensor of shape %A as shape %A" t.Shape shape
        let result = Array.copy t.Values
        t.CreateShaped(result, shape)

    override t.Cast(dtype: DType) =
        if dtype = t.DType then 
            upcast t
        else 
            RawTensor.Create(t.ToArray(), dtype=dtype, backend=t.Backend, device=t.Device)

    member t.ToInt32() = t.Cast(Int32)
    member t.ToFloat32() = t.Cast(Float32)
    member t.ToFloat64() = t.Cast(Float64)

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

    let inline Create ofFloat32 ofDouble ofInt (value:obj) : (^T[] * int[]) = 
        let values, shape = value |> flatArrayAndShape<float32>
        if notNull values then 
            (values |> Array.map ofFloat32, shape)
        else 
            let values, shape = value |> flatArrayAndShape<double>
            if notNull values then 
                (values |> Array.map ofDouble, shape)
            else
                let values, shape = value |> flatArrayAndShape<int>
                if notNull values then 
                    (values |> Array.map ofInt, shape)
                else
                    invalidArg "value" "Cannot convert value to RawTensorCPU"

    let inline CompareTo(t1: RawTensorCPU< ^T >, t2: RawTensor) =
        NonStructuralComparison.compare (t1.ToValue() :?> ^T ) (t2.ToValue() :?> ^T )

    let inline RandomMultinomial ofInt (t: RawTensorCPU< ^T >, numSamples) : (^T[] * int[]) =
        if t.Dim < 1 || t.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" t.Shape
        if t.Dim = 1 then
            let p = t.Values |> Array.map float
            let result = [|for i=0 to numSamples-1 do yield ofInt (Random.ChoiceIndex(p))|]
            (result, [|numSamples|])
        else
            let p = t.ToArray() :?> float32[,] |> Array2D.map float
            let d1 = p.GetLength(0)
            let result = [| for i in 0 .. (d1 * numSamples - 1) -> ofInt (Random.ChoiceIndex(p.[(i%numSamples),*])) |]
            (result, [| d1; numSamples |]) 

    let inline Equals(t1: RawTensorCPU< ^T >, t2: RawTensor) = 
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    let inline ApproximatelyEquals(t1: RawTensorCPU< ^T >, t2:RawTensor, tolerance: ^T) =
        match t2 with
        | :? RawTensorCPU< ^T > as t2 -> t1.Shape = t2.Shape && arraysApproximatelyEqual tolerance t1.Values t2.Values
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

    let inline internal MaxIndexT(t: RawTensorCPU< ^T >) =
        t.FlatIndexToIndex(maxIndex t.Values)

    let inline internal MinIndexT(t: RawTensorCPU< ^T >) =
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
        if not (shapeContains t1.Shape t2.Shape) then failwithf "Expecting t1.Shape to contain t2.Shape, received %A, %A" t1.Shape t2.Shape
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
        if t1.Dim <> 2 || t2.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        if t1cols <> t2rows then invalidOp <| sprintf "Cannot multiply Tensors with shapes %A, %A" t1.Shape t2.Shape
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

    let inline TransposeT2(t: RawTensorCPU< ^T >) : (^T[] * int[]) =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let trows = t.Shape.[0]
        let tcols = t.Shape.[1]
        let result = Array.zeroCreate (tcols*trows)
        let vs = t.Values
        for i in 0 .. tcols - 1 do 
            for j in 0 .. trows - 1 do 
                result.[i*trows + j] <- vs.[j*tcols + i]
        (result, [| tcols; trows |])

    let inline Conv1D 
           // type-specific witness used to create tensors of the natural implementation type
           (createZeroTensor: int[] -> (^TensorImpl :> RawTensorCPU< ^T >)) 
           // the actual parameters
           (t1: RawTensorCPU< ^T >, t2: RawTensor, stride, padding) : ^TensorImpl =

        // t1: input, NxCxI (batchSize x inputChannels, inputLength)
        // t2: filters, KxCxF (outputChannels x inputChannels, kernelLength)
        if t1.Dim <> 3 || t2.Dim <> 3 then invalidOp <| sprintf "Expecting two 3d Tensors t1, t2 where t1 = input: NxCxI (batchSize x inputChannels, inputLength) and filters: KxCxF (outputChannels x inputChannels, kernelLength), received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1 =
            if padding = 0 then
                t1
            elif padding > 0 then
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding * 2
                let t = createZeroTensor(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorCPU< ^T >
            else
                invalidOp <| sprintf "Expecting padding >= 0, received %A" padding
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputLength = t1.Shape.[2]
        let outputChannels = t2.Shape.[0]
        if t2.Shape.[1] <> inputChannels then invalidOp <| sprintf "Input and filters have different num_channels: %A, %A" inputChannels t2.Shape.[1]
        let kernelLength = t2.Shape.[2]
        if kernelLength > inputLength then invalidOp <| sprintf "Expecting kernelLength <= inputLength, received %A, %A" kernelLength inputLength
        let outputLength = inputLength - kernelLength + 1
        let outputShape = [|batchSize; outputChannels; outputLength|]
        let result = createZeroTensor(outputShape)
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
        elif stride > 1 then
            let outputLength = (float outputLength) / (float stride) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputLength|]
            let mutable sresult = createZeroTensor(outputShape)
            for v=0 to outputLength-1 do
                let sliceBounds = array2D [[0; batchSize-1]; [0; outputChannels-1]; [v * stride; v * stride]]
                let slice = result.GetSlice(sliceBounds).UnsqueezeT(2)
                sresult <- sresult.AddTTSlice([|0; 0; v|], slice) :?> ^TensorImpl
            sresult 
        else
            invalidOp <| sprintf "Expecting stride >= 1, received %A" stride

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
    static member Random(shape:int[])  = RawTensorCPU.Random float32 shape |> RawTensorFloat32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal float32 shape |> RawTensorFloat32CPU
    static member Create(value:obj) = RawTensorCPU.Create float32 float32 float32 (value) |> RawTensorFloat32CPU
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat32CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat32CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat32CPU.Zeros(shape)
    override t.One() = create([|1.f|], [||])
    override t.Ones(shape) = upcast RawTensorFloat32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial float32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPU.ApproximatelyEquals(t1, t2, float32 tolerance)
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
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D RawTensorFloat32CPU.Zeros (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.TransposeT2() = RawTensorCPU.TransposeT2(t) |> create
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
    static member Random(shape:int[])  = RawTensorCPU.Random double shape |> RawTensorFloat64CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal double shape |> RawTensorFloat64CPU
    static member Create(value:obj) = RawTensorCPU.Create double double double (value) |> RawTensorFloat64CPU
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat64CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat64CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat64CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat64CPU([|1.0|], [||])
    override t.Ones(shape) = upcast RawTensorFloat64CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat64CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial double (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPU.ApproximatelyEquals(t1, t2, double tolerance)
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
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D RawTensorFloat64CPU.Zeros (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.TransposeT2() = RawTensorCPU.TransposeT2(t) |> create
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
    override __.Random(shape:int[]) = upcast RawTensorFloat64CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat64CPU.Create(values)

type RawTensorInt32CPU(values: int32[], shape:int[]) =
    inherit RawTensorCPU<int32>(values, shape, Int32)

    static let create(values, shape) : RawTensor = upcast RawTensorInt32CPU(values, shape)
    static let createf(values, shape) : RawTensor = upcast RawTensorFloat32CPU(values, shape)
    static member Zero() = RawTensorCPU.Zero() |> RawTensorInt32CPU
    static member One() = RawTensorCPU.One() |> RawTensorInt32CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorInt32CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorInt32CPU
    static member Random(shape:int[])  = RawTensorCPU.Random int32 shape |> RawTensorInt32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal int32 shape |> RawTensorInt32CPU
    static member Create(value:obj) = RawTensorCPU.Create int32 int32 int32 (value) |> RawTensorInt32CPU
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt32CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt32CPU.Create(values)
    override t.Zero() = upcast RawTensorInt32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt32CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt32CPU([|1|], [||])
    override t.Ones(shape) = upcast RawTensorInt32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPU.ApproximatelyEquals(t1, t2, int32 tolerance)
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
    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D RawTensorInt32CPU.Zeros (t1, t2, stride, padding) :> _
    override t.NegT() = RawTensorCPU.NegT(t) |> create
    override t.SumT() = RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
    override t.TransposeT2() = RawTensorCPU.TransposeT2(t) |> create
    override t.SignT() = RawTensorCPU.SignT int32 t |> create
    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

    member t.ToFloat32() = base.ToFloat32() :?> RawTensorFloat32CPU

    // Note, these produce Float32 tensors implicitly
    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1.ToFloat32(), t2) |> createf
    override t1.PowT0T(t2) = RawTensorCPU.PowT0T(t1.ToFloat32(), t2) |> createf
    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1.ToFloat32(), t2) |> createf
    override t.FloorT() = RawTensorCPU.FloorT(t.ToFloat32()) |> createf
    override t.CeilT() = RawTensorCPU.CeilT(t.ToFloat32()) |> createf
    override t.RoundT() = RawTensorCPU.RoundT(t.ToFloat32()) |> createf
    override t.SigmoidT() = RawTensorCPU.SigmoidT(t.ToFloat32()) |> createf
    override t.ExpT() = RawTensorCPU.ExpT(t.ToFloat32()) |> createf
    override t.LogT() = RawTensorCPU.LogT(t.ToFloat32()) |> createf
    override t.Log10T() = RawTensorCPU.Log10T(t.ToFloat32()) |> createf
    override t.SqrtT() = RawTensorCPU.SqrtT(t.ToFloat32()) |> createf
    override t.SinT() = RawTensorCPU.SinT(t.ToFloat32()) |> createf
    override t.CosT() = RawTensorCPU.CosT(t.ToFloat32()) |> createf
    override t.TanT() = RawTensorCPU.TanT(t.ToFloat32()) |> createf
    override t.SinhT() = RawTensorCPU.SinhT(t.ToFloat32()) |> createf
    override t.CoshT() = RawTensorCPU.CoshT(t.ToFloat32()) |> createf
    override t.TanhT() = RawTensorCPU.TanhT(t.ToFloat32()) |> createf
    override t.AsinT() = RawTensorCPU.AsinT(t.ToFloat32()) |> createf
    override t.AcosT() = RawTensorCPU.AcosT(t.ToFloat32()) |> createf
    override t.AtanT() = RawTensorCPU.AtanT(t.ToFloat32()) |> createf

and RawTensorInt32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt32CPU.Zero()
    override __.One = upcast RawTensorInt32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt32CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorInt32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt32CPU.Create(values)

                