namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open TorchSharp
open TorchSharp.Tensor

type TorchShape = int64[]

[<AutoOpen>]
module internal Utils = 

    let int64s (b: int[]) = Array.map int64 b

    let toTorchType dtype =
        match dtype with 
        | Dtype.Bool -> ScalarType.Bool
        | Dtype.Int8 -> ScalarType.SByte
        | Dtype.Byte -> ScalarType.Byte
        | Dtype.Int16 -> ScalarType.Short
        | Dtype.Int32 -> ScalarType.Int
        | Dtype.Int64 -> ScalarType.Long
        | Dtype.Float32 -> ScalarType.Float
        | Dtype.Float64 -> ScalarType.Double

    let fromTorchType ttype =
        match ttype with 
        | ScalarType.Bool -> Dtype.Bool
        | ScalarType.SByte -> Dtype.Int8
        | ScalarType.Byte -> Dtype.Byte
        | ScalarType.Short -> Dtype.Int16
        | ScalarType.Int -> Dtype.Int32
        | ScalarType.Long -> Dtype.Int64
        | ScalarType.Float -> Dtype.Float32
        | ScalarType.Double -> Dtype.Float64
        |  _ -> failwith "fromTorchType - other type"

    let toTorchShape (shape: Shape) : TorchShape = int64s shape

    let fromTorchShape (shape: int64[]) = shape |> Array.map int

    type Device with 
        member x.TorchDeviceType : TorchSharp.DeviceType = enum (int x.DeviceType)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    let torchMoveTo (tt: TorchTensor) (device: Device) =
        tt.ToDevice(device.TorchDeviceType, device.DeviceIndex)

    type RawTensor with
        member x.TorchTensor = (x :?> TorchRawTensor).TorchTensor

/// This is the base class for all RawTensorXyz tuypes.
/// All type-independent operations are implemented directly on this class. 
type TorchRawTensor(tt: TorchTensor, shape: Shape, dtype: Dtype, device: Device) =

    inherit RawTensor()

    // Note, shape and dtype are stored as fields. These dupicate information in TorchTensor, but
    // it is a little too costly to repeatedly re-extract this information.
    //
    // 'device' is not stored as a field, it is rarely accessed and can be fetched from TorchTensor

#if DEBUG
    // Check the invariants associated with the tensors
    do 
       if tt.Type <> toTorchType dtype then
           failwithf "mismatched Torch tensor type, expected %A, got %A" (toTorchType dtype) tt.Type

       if int tt.DeviceType <> int device.DeviceType then
           failwithf "mismatched Torch tensor device, expected %A, got %A" tt.DeviceType device.DeviceType

       if int tt.DeviceIndex <> int device.DeviceIndex then
           failwithf "mismatched Torch tensor index, expected %A, got %A" tt.DeviceIndex device.DeviceIndex

       if toTorchShape shape <> tt.Shape then 
           failwithf "mismatched Torch tensor shape, expected %A, got %A" (toTorchShape shape) tt.Shape

    let device = () // make sure 'device' isn't accessed in a member and stored as a field
#endif
    let mutable tt = tt
    let mutable isMutable = false
    let checkMutable() = if not isMutable then failwith "the tensor can't be mutated" 
    do ignore device

    override _.Shape = shape
    override _.Dim = shape.Length
    override _.Nelement = shapeLength shape
    override _.Dtype = dtype
    override _.DeviceType : DiffSharp.DeviceType = enum (int tt.DeviceType)
    override t.Device = Device(t.DeviceType, tt.DeviceIndex)
    override _.Backend = Backend.Torch
    override _.Handle = box tt

    member t.MakeLike(tt, ?shape, ?dtype, ?device) : RawTensor =
        upcast TorchRawTensor(tt, defaultArg shape t.Shape, defaultArg dtype t.Dtype, defaultArg device t.Device)

    member x.TorchTensor = tt

    override t.GetSlice(fullBounds:int[,]) =
        let newShape = Shape.checkCanGetSlice t.Shape fullBounds
        let mutable res = tt
        let mutable dim = 0 
        for i=0 to (fullBounds.GetLength(0) - 1) do
            let start = fullBounds.[i,0]
            let stop = fullBounds.[i,1] + 1

            let len = stop - start
            use idxs = LongTensor.Arange((int64 start).ToScalar(), (int64 stop).ToScalar(), 1L.ToScalar(), tt.DeviceType, tt.DeviceIndex)
            res <- res.IndexSelect(int64 dim, idxs)  // yield len // if len=1 then squeeze this dimension
            if fullBounds.[i, 2] = 1 && len = 1 then 
                res <- res.Squeeze(int64 dim)  // yield len // if len=1 then squeeze this dimension
            else
                dim <- dim + 1
        t.MakeLike(tt=res, shape=newShape)

    override t.Clone() =
        t.MakeLike(tt.Clone())

    override t.ComputeHash() = 
        // Torch Tensors must be CPU before DataItem can be accessed
        let tt = torchMoveTo tt Device.CPU

        let shape = t.Shape
        let mutable res = hash shape
        let n = shapeLength shape
        match dtype with 
        | Dtype.Int8 ->
            let data = tt.Data<sbyte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | Dtype.Byte ->
            let data = tt.Data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | Dtype.Bool ->
            let data = tt.Data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | Dtype.Int16 ->
            let data = tt.Data<int16>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i] )
        | Dtype.Int32 ->
            let data = tt.Data<int32>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | Dtype.Int64 -> 
            let data = tt.Data<int64>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | Dtype.Float32 ->
            let data = tt.Data<single>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        | Dtype.Float64 ->
            let data = tt.Data<double>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        res
    
    override t.Expand(newShape) =
        t.MakeLike(tt.Expand(toTorchShape newShape), shape=newShape)

    override t.GetItem(indexes) = 

        let item = 
            match indexes with 
            | [| |] -> tt
            | [| i0 |] -> tt.[int64 i0]
            | [| i0; i1 |] -> tt.[int64 i0, int64 i1]
            | [| i0; i1; i2 |] -> tt.[int64 i0, int64 i1, int64 i2]
            | [| i0; i1; i2; i3 |] -> tt.[int64 i0, int64 i1, int64 i2, int64 i3]
            | _ -> 
                let shape = t.Shape
                tt.View(toTorchShape [|shapeLength shape|]).[int64 (indexToFlatIndex shape indexes)]

        // Torch Tensors must be CPU before DataItem can be accessed
        let item = torchMoveTo item Device.CPU

        let obj = 
            match dtype with 
            | Dtype.Bool -> box (item.DataItem<bool>())
            | Dtype.Byte -> box (item.DataItem<byte>())
            | Dtype.Int8 -> box (item.DataItem<int8>())
            | Dtype.Int16 -> box (item.DataItem<int16>())
            | Dtype.Int32 -> box (item.DataItem<int32>())
            | Dtype.Int64 -> box (item.DataItem<int64>())
            | Dtype.Float32 -> box (item.DataItem<float32>())
            | Dtype.Float64 -> box (item.DataItem<double>())
        obj

    member t.ToValuesTyped<'T, 'T2>(conv) : obj =
        // Torch Tensors must be CPU before DataItem can be accessed
        let tt = torchMoveTo tt Device.CPU

        match t.Shape with
        | [|  |] -> t.GetItem()
        | [| d0 |] -> upcast Array.init<'T> d0 (fun i -> tt.[int64 i].DataItem<'T2>() |> conv)
        | [| d0; d1 |] -> upcast Array2D.init<'T> d0 d1 (fun i j -> tt.[int64 i, int64 j].DataItem<'T2>() |> conv)
        | [| d0; d1; d2 |]  -> upcast Array3D.init<'T> d0 d1 d2 (fun i j k -> tt.[int64 i, int64 j, int64 k].DataItem<'T2>() |> conv)
        | [| d0; d1; d2; d3 |]  -> upcast Array4D.init<'T> d0 d1 d2 d3 (fun i j k l -> tt.[int64 i, int64 j, int64 k, int64 l].DataItem<'T2>() |> conv)
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override t.ToValues() =
        
        match dtype with 
        | Dtype.Bool -> t.ToValuesTyped<bool, bool>(id)
        | Dtype.Byte -> t.ToValuesTyped<byte, byte>(id)
        | Dtype.Int8 -> t.ToValuesTyped<sbyte, sbyte>(sbyte)
        | Dtype.Int16 -> t.ToValuesTyped<int16, int16>(id)
        | Dtype.Int32 -> t.ToValuesTyped<int32, int32>(id)
        | Dtype.Int64 -> t.ToValuesTyped<int64, int64>(id)
        | Dtype.Float32 -> t.ToValuesTyped<float32, float32>(id)
        | Dtype.Float64 -> t.ToValuesTyped<double, double>(id)

    member _.ToRawData<'T>() : 'T[] =
        // Torch Tensors must be CPU before raw data can be accessed
        let tt2 = torchMoveTo tt Device.CPU

        let data = tt2.Data<'T>()
        let res = Array.zeroCreate<'T> (int32 tt2.NumberOfElements)
        for i in 0 .. int32 tt2.NumberOfElements - 1 do
            res.[i] <- data.[i]
        res

    member t.ToRawData() =
        match dtype with 
        | Dtype.Bool -> t.ToRawData<bool>() |> box
        | Dtype.Byte -> t.ToRawData<byte>() |> box
        | Dtype.Int8 -> t.ToRawData<sbyte>() |> box
        | Dtype.Int16 -> t.ToRawData<int16>() |> box
        | Dtype.Int32 -> t.ToRawData<int32>() |> box
        | Dtype.Int64 -> t.ToRawData<int64>() |> box
        | Dtype.Float32 -> t.ToRawData<float32>() |> box
        | Dtype.Float64 -> t.ToRawData<double>() |> box

    override _.StackTs(tensors, dim) =
        let tts, shapes = tensors |> Array.map (fun t -> (t :?> TorchRawTensor).TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _shape2, newShape = Shape.checkCanStack shapes dim
        let result = tts.Stack(int64 dim)
        (tensors.[0] :?> TorchRawTensor).MakeLike(result, newShape)

    override t.UnstackT(dim) = 
        let shape = t.Shape
        let _shape1, _shape2, unstackedShape = Shape.checkCanUnstack shape dim
        let results = tt.Unbind(dim)
        results |> Array.map (fun rvalues -> t.MakeLike(rvalues, shape=unstackedShape))

    override t.CatTs(tensors, dim) = 
        let values, shapes = tensors |> Array.map (fun t -> t.TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _m2, _shape3, outShape = Shape.checkCanCat shapes dim
        let result = values.Cat(int64 dim)
        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        let results = tt.SplitWithSizes(int64s sizes, dim)
        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, shape=outShape))

    override t.TransposeT(dim0, dim1) =
        Shape.checkCanTranspose t.Shape dim0 dim1
        let result = tt.Transpose(int64 dim0, int64 dim1)
        let shape = result.Shape |> Array.map int32
        t.MakeLike(result, shape=shape)

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        let newShape = Shape.computeTranspose2d t.Shape
        let result = tt.T()
        t.MakeLike(result, shape=newShape)

    override t.SqueezeT(dim) = 
        let shape = t.Shape
        let newShape = Shape.squeeze dim shape
        let mutable res = tt
        let mutable c = 0
        for i in 0 .. t.Dim - 1 do
            if shape.[i] = 1 && (dim = -1 || i = dim) then 
                res <- res.Squeeze(int64 c)
            else   
                c <- c + 1
        t.MakeLike(res, shape=newShape)

    override t.UnsqueezeT(dim) = 
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        t.MakeLike(tt.Unsqueeze(int64 dim), shape=outputShape)

    override t.FlipT(dims:int[]) = 
        // "flip_cuda" not implemented for 'Bool'"
        let result =
            if dtype = Dtype.Bool then 
                tt.ToType(ScalarType.Byte).Flip(int64s dims).ToType(ScalarType.Bool)
            else
                tt.Flip(int64s dims)
        t.MakeLike(result)

    override t.DilateT(dilations:int[]) = 
        Shape.checkCanDilate t.Dim dilations
        let outputShape = Shape.dilated t.Shape dilations
        let dims = dilations.Length
        let mutable res = tt
        for i=0 to dims-1 do
            let s = res.Shape
            s.[i] <- int64 outputShape.[i]
            let resnew = t.ZerosLike(fromTorchShape s)
            let indices = Array.init t.Shape.[i] id |> Array.map ((*) dilations.[i] >> int64)
            let mutable d = TorchInt64TensorOps().CreateFromFlatArray(indices, shape=[|t.Shape.[i]|], device=t.Device)
            for _=0 to i-1 do
                d <- d.UnsqueezeT(0)
            for _=i+1 to dims-1 do
                d <- d.UnsqueezeT(d.Dim)
            d <- d.Expand(fromTorchShape res.Shape)
            res <- resnew.TorchTensor.Scatter(int64 i, d.TorchTensor, res)
        t.MakeLike(res, outputShape)

    override t.UndilateT(dilations:int[]) =
        let shape = t.Shape
        let outputShape = Shape.undilatedShape shape dilations
        let mutable res = tt
        for d in 0 .. dilations.Length - 1 do
            res <- res.Slice(int64 d, 0L, int64 shape.[d], int64 dilations.[d])
        t.MakeLike(res, outputShape)

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype
        if indices.Dtype <> Dtype.Int32 then opNotSupported "Gather (indices must currently be int32 tensors in DiffSharp" indices.Dtype

        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = indices.Cast(Dtype.Int64)
        let res = t.TorchTensor.Gather(int64 dim, indices.TorchTensor)
        t.MakeLike(res, indices.Shape)

    override t.ViewT(shape:Shape) =
        Shape.checkCanView t.Shape shape
        t.MakeLike(tt.Reshape(toTorchShape shape), shape=shape)  // Use Reshape instead of View to ensure underlying non-contiguous libtorch tensors can be viewed. Internally Reshape uses View if possible, otherwise it copies data to a contiguous tensor and then views.

    override t.Cast(newDtype: Dtype) =
        if newDtype = dtype then 
            upcast t
        else 
            let result = tt.ToType(toTorchType newDtype)
            t.MakeLike(result, dtype=newDtype)

    override t.MoveTo(device: Device) =
        if t.Device = device then (t :> _) else
        let tt2 = torchMoveTo tt device
        t.MakeLike(tt2, device=device)

    override t.Equals(t2:RawTensor) : bool = 
        if dtype = t2.Dtype then
            let r1 = (t.Shape = t2.Shape)
            if not r1 then false else
            let tt2 = t2.TorchTensor
            let r2 = tt.Equal(tt2)
            r2
        else 
            opNotSupported2 "Equals" dtype t2.Dtype

    override t.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        if dtype = t2.Dtype then
            match dtype with 
            | Dtype.IntegralOrBool -> t.Equals(t2)
            | _ -> tt.AllClose(t2.TorchTensor, relativeTolerance, absoluteTolerance)
        else 
            opNotSupported2 "Equals" dtype t2.Dtype

    override t.ClampT(low, high) =
        let result = tt.Clamp(low.TorchTensor.Item(), high.TorchTensor.Item())
        t.MakeLike(result)

    override t1.LtTT(t2) =
        let result = tt.Lt(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.GtTT(t2) =
        let result = tt.Gt(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.LeTT(t2) = 
        let result = tt.Le(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.GeTT(t2) = 
        let result = tt.Ge(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.EqTT(t2) = 
        let result = tt.Eq(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.NeqTT(t2) = 
        let result = tt.Ne(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t.MaxIndexT() = 

        let res = Array.zeroCreate<int64> t.Dim
        let idxs = Array.zeroCreate t.Dim
        let mutable values = tt
        for i = t.Dim - 1 downto 0 do 
            let (struct (values2, indexes)) = values.Max(int64 i)
            values <- values2
            idxs.[i] <- indexes
        for i = 0 to t.Dim - 1 do 
            let idx = idxs.[i]

            // Torch Tensors must be CPU before DataItem can be accessed
            let idx = torchMoveTo idx Device.CPU

            res.[i] <- 
                match i with 
                | 0 -> idx.DataItem<int64>()
                | 1 -> idx.[res.[0]].DataItem<int64>() 
                | 2 -> idx.[res.[0], res.[1]].DataItem<int64>() 
                | 3 -> idx.[res.[0], res.[1], res.[2]].DataItem<int64>() 
                | _ -> failwith "MaxIndexT > 4d nyi for torch"
        res |> Array.map int32

    // TODO: use Torch min operation
    override t.MinIndexT() = 
        match dtype with 
        | Dtype.Bool -> t.Cast(Dtype.Int8).MinIndexT() // TODO: could likely be improved
        | _ -> t.NegT().MaxIndexT()

    override t1.AddTT(t2) =
        let result = tt.Add(t2.TorchTensor) 
        t1.MakeLike(result)

    override t1.AddTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Add(t2v) 
        t1.MakeLike(result)

    override t1.AddT2T1(t2) = 
        let result = tt.Add(t2.TorchTensor) 
        t1.MakeLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = Shape.unsqueezeAs shape2 shape1
        let t2Expanded = t2.TorchTensor.Expand(toTorchShape expandedShape2)
        let res = tt.Clone()
        let mutable t1Slice = res // will share memory with res
        for d in 0 .. location.Length - 1 do 
            let len2 = expandedShape2.[d]
            if location.[d] <> 0 || len2 <> shape1.[d] then 
                t1Slice <- t1Slice.Narrow(int64 d, int64 location.[d], int64 len2)
        t1Slice.AddInPlace(t2Expanded) |> ignore
        t1.MakeLike(res)

    override t1.SubTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "SubT" dtype t2.Dtype
        | _ ->
        let result = tt.Sub(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.SubT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v - t2.TorchTensor
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.SubTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Sub(t2v)
        t1.MakeLike(result)

    override t1.MulTT(t2) = 
        let result = tt.Mul(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.MulTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Mul(t2v)
        t1.MakeLike(result)

    override t1.DivTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "DivTT" dtype t2.Dtype
        | _ ->
        let result = tt.Div(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.DivT0T(t2) =
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "DivTT" dtype t2.Dtype
        | _ ->
        let t1v = t1.TorchTensor.Item()
        let result = t1v / t2.TorchTensor
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.DivTT0(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "DivTT" dtype t2.Dtype
        | _ ->
        let t2v = t2.TorchTensor.Item()
        let result = tt.Div(t2v)
        t1.MakeLike(result)

    override t1.PowTT(t2) =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowTT" dtype
        | _ -> 
        let result = tt.Pow(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.PowT0T(t2) = 
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowT0T" dtype
        | _ -> 
        let result = t1.Expand(t2.Shape).TorchTensor.Pow(t2.TorchTensor)
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.PowTT0(t2) =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowTT0" dtype
        | _ -> 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Pow(t2v)
        t1.MakeLike(result)

    override t1.MatMulTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "MatMulTT" dtype t2.Dtype
        | _ ->  
        let _, _ = Shape.checkCanMatmul t1.Shape t2.Shape
        let result =
            // "addmm for CUDA tensors only supports floating-point types. Try converting the tensors with .float()" | const char *
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                let tt1 = tt.ToType(ScalarType.Double)
                let tt2 = t2.TorchTensor.ToType(ScalarType.Double)
                tt1.Mm(tt2).Round().ToType(toTorchType dtype) 
            | _ ->
                tt.Mm(t2.TorchTensor)
        t1.MakeLike(result, [| t1.Shape.[0]; t2.Shape.[1] |])

    override t1.Conv1D(t2, stride, padding) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelSize, _outputChannels, _outputSize, outputShape =
            Shape.checkCanConv1d t1.DeviceType t2.DeviceType dtype t2.Dtype t1.Shape t2.Shape stride padding 1
        let resultt =
            // "conv1d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                tt.ToType(ScalarType.Double).Conv1D(t2.TorchTensor.ToType(ScalarType.Double), stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L)).Round().ToType(toTorchType dtype) 
            | _ ->
                tt.Conv1D(t2.TorchTensor, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L))
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv2D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanConv2d t1.DeviceType t2.DeviceType dtype t2.Dtype t1.Shape t2.Shape strides paddings [| 1;1 |]
        let resultt =
            // "conv2d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                tt.ToType(ScalarType.Double).Conv2D(t2.TorchTensor.ToType(ScalarType.Double), strides=int64s strides, padding=int64s paddings).Round().ToType(toTorchType dtype) 
            | _ ->
                tt.Conv2D(t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv3D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanConv3d t1.DeviceType t2.DeviceType dtype t2.Dtype  t1.Shape t2.Shape strides paddings [| 1;1;1 |]
        let resultt =
            // "conv2d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                tt.ToType(ScalarType.Double).Conv3D(t2.TorchTensor.ToType(ScalarType.Double), strides=int64s strides, padding=int64s paddings).Round().ToType(toTorchType dtype) 
            | _ ->
                tt.Conv3D(t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxPool1D(kernelSize, stride, padding) = 
        let _batchSize, _channels, _inputSize, _outputSize, outputShape =
            Shape.checkCanMaxpool1d dtype t1.Shape kernelSize stride padding
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "MaxPool1D" dtype
        | _ ->
        let struct (resultt, indicest) = tt.MaxPool1DWithIndices(int64 kernelSize, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L))
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool2D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanMaxpool2d dtype t1.Shape kernelSize strides paddings
        let struct (resultt, indicest) = tt.MaxPool2DWithIndices(int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool3D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanMaxpool3d dtype t1.Shape kernelSize strides paddings
        let struct (resultt, indicest) = tt.MaxPool3DWithIndices(int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxUnpool1D(indices, outputSize) = 
        // NOTE: LibTorch has no torch::max_unpool1d and so TorchSharp has Tensor.MaxUnpool1D
        // So use MaxUnpool2D instead
        //let batchSize, channels, _inputSize, _outputShape = Shape.computeMaxUnpool1d t1.Shape outputSize
        let t1X = t1.UnsqueezeT(2)
        let indicesX = indices.UnsqueezeT(2)
        let resulttX = t1X.MaxUnpool2D(indicesX, [| outputSize.[0]; outputSize.[1]; 1; outputSize.[2] |])
        let resultt = resulttX.SqueezeT(2)
        resultt

    override t1.MaxUnpool2D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape =
            Shape.checkCanMaxunpool2d dtype t1.Shape indices.Dtype indices.Shape outputSize
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(Dtype.Int64)

        // note, LibTorch only wants the last two elements of the output size passsed in
        // "There should be exactly two elements (height, width) in output_size (max_unpooling2d_shape_check at ...)"
        let outputSize = outputSize.[2..3]
        
        // TODO: consider switching to the torch::nn module for MaxUnpool2d

        let resultt = tt.MaxUnpool2D(indices.TorchTensor, int64s outputSize)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxUnpool3D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape =
            Shape.checkCanMaxunpool3d dtype t1.Shape indices.Dtype indices.Shape outputSize
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(Dtype.Int64)

        // note, LibTorch only wants the last three elements of the output size passsed in
        // "There should be exactly three elements (depth, height, width) in output_size (max_unpooling3d_shape_check at ..\..\aten\src\ATen\native\MaxUnpooling.cpp:231)"
        let outputSize = outputSize.[2..4]
        
        // NOTE: strides and padding must always be specified for torch::max_unpool3d C++ entry
        // TODO: consider switching to the torch::nn module for MaxUnpool
        let strides = outputSize |> Array.map (fun _ -> 1L)
        let padding = outputSize |> Array.map (fun _ -> 0L)
        let resultt = tt.MaxUnpool3D(indices.TorchTensor, int64s outputSize, strides, padding)
        t1.MakeLike(resultt, shape=outputShape)

    override t.SumT2Dim0() =
        let result = tt.Sum([| 0L |], ``type``= Nullable(tt.Type))
        let resultShape = [|t.Shape.[1]|]
        t.MakeLike(result, shape=resultShape)

    override t.NegT() =
        match dtype with 
        | Dtype.Bool -> opNotSupported "NegT" dtype
        | _ ->  t.MakeLike(-tt)

    override t.SumT(?resultType) =
        let typeArg = match resultType with None -> Nullable() | Some dt -> Nullable(toTorchType dt)
        let outType = match resultType with None -> dtype.SummationType | Some dt -> dt
        t.MakeLike(tt.Sum(typeArg), shape=Shape.scalar, dtype=outType)

    override t.SignT() =
        t.MakeLike(tt.Sign())

    override t.FloorT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "FloorT" dtype
        | _ ->  t.MakeLike(tt.Floor())

    override t.CeilT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CeilT" dtype
        | _ ->  t.MakeLike(tt.Ceil())

    override t.RoundT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "RoundT" dtype
        | _ ->  t.MakeLike(tt.Round())

    override t.AbsT() = 
        match dtype with 
        | Dtype.Bool -> opNotSupported "AbsT" dtype
        | Dtype.Int8 -> t.Cast(Dtype.Int32).AbsT().Cast(Dtype.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ -> t.MakeLike(tt.Abs ())

    override t.SoftplusT() = 
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SoftplusT" dtype
        | _ -> t.MakeLike(tt.Softplus())

    override t.ReluT() =
        match dtype with 
        | Dtype.Bool -> opNotSupported "ReluT" dtype
        | Dtype.Int8 -> t.Cast(Dtype.Int32).ReluT().Cast(Dtype.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ ->   t.MakeLike(tt.Relu())

    override t.SigmoidT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SigmoidT" dtype
        | _ ->  t.MakeLike(tt.Sigmoid())

    override t.ExpT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "ExpT" dtype
        | _ ->  t.MakeLike(tt.Exp())

    override t.LogT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "LogT" dtype
        | _ ->  t.MakeLike(tt.Log())

    override t.Log10T() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "Log10T" dtype
        | _ ->   t.MakeLike(tt.Log10())

    override t.SqrtT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SqrtT" dtype
        | _ ->  t.MakeLike(tt.Sqrt())

    override t.SinT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SinT" dtype
        | _ ->  t.MakeLike(tt.Sin())

    override t.CosT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CosT" dtype
        | _ ->  t.MakeLike(tt.Cos())

    override t.TanT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "TanT" dtype
        | _ ->  t.MakeLike(tt.Tan())

    override t.SinhT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SinhT" dtype
        | _ ->  t.MakeLike(tt.Sinh())

    override t.CoshT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CoshT" dtype
        | _ ->  t.MakeLike(tt.Cosh())

    override t.TanhT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "TanhT" dtype
        | _ ->  t.MakeLike(tt.Tanh())

    override t.AsinT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AsinT" dtype
        | _ ->  t.MakeLike(tt.Asin())

    override t.AcosT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AcosT" dtype
        | _ ->  t.MakeLike(tt.Acos())

    override t.AtanT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AtanT" dtype
        | _ ->  t.MakeLike(tt.Atan())

    new (info: System.Runtime.Serialization.SerializationInfo, _context: System.Runtime.Serialization.StreamingContext) =
        let dtype = info.GetValue("dtype", typeof<Dtype>) :?> Dtype
        let shape = info.GetValue("shape", typeof<Shape>) :?> Shape
        let tt =
            match dtype with 
            | Dtype.Bool -> 
                let data = info.GetValue("data", typeof<bool[]>)  :?> bool[]
                BoolTensor.From (data, toTorchShape shape) 
            | Dtype.Byte -> 
                let data = info.GetValue("data", typeof<byte[]>)  :?> byte[]
                ByteTensor.From (data, toTorchShape shape) 
            | Dtype.Int8 -> 
                let data = info.GetValue("data", typeof<sbyte[]>)  :?> sbyte[]
                SByteTensor.From (data, toTorchShape shape) 
            | Dtype.Int16 -> 
                let data = info.GetValue("data", typeof<int16[]>)  :?> int16[]
                ShortTensor.From (data, toTorchShape shape) 
            | Dtype.Int32 -> 
                let data = info.GetValue("data", typeof<int32[]>)  :?> int32[]
                IntTensor.From (data, toTorchShape shape) 
            | Dtype.Int64 -> 
                let data = info.GetValue("data", typeof<int64[]>)  :?> int64[]
                LongTensor.From (data, toTorchShape shape) 
            | Dtype.Float32 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                FloatTensor.From (data, toTorchShape shape) 
            | Dtype.Float64 -> 
                let data = info.GetValue("data", typeof<double[]>)  :?> double[]
                DoubleTensor.From (data, toTorchShape shape) 

        TorchRawTensor(tt, shape, dtype, Device.CPU)

    interface System.Runtime.Serialization.ISerializable with

        //[SecurityPermissionAttribute(SecurityAction.Demand,  SerializationFormatter = true)]
        member t.GetObjectData(info, _context) =
            
            // Torch Tensors must be CPU before they can access RawData
            let tCpu = t.MoveTo(Device.CPU) :?> TorchRawTensor

            info.AddValue("dtype", t.Dtype)
            info.AddValue("shape", t.Shape)
            info.AddValue("data", tCpu.ToRawData())


    override _.ClampInPlace(low, high) = 
        // TODO - next version of TorchSharp will have in place version of this
        checkMutable()
        tt <- tt.Clamp(low.TorchTensor.Item(), high.TorchTensor.Item())

    override _.LtInPlace(t2) = checkMutable(); tt.LtInPlace(t2.TorchTensor) |> ignore

    override _.GtInPlace(t2) = checkMutable(); tt.GtInPlace(t2.TorchTensor) |> ignore

    override _.LeInPlace(t2) = checkMutable(); tt.LeInPlace(t2.TorchTensor) |> ignore

    override _.GeInPlace(t2) = checkMutable(); tt.GeInPlace(t2.TorchTensor) |> ignore

    override _.EqInPlace(t2) = checkMutable(); tt.EqInPlace(t2.TorchTensor) |> ignore

    override _.NeqInPlace(t2) = checkMutable(); tt.NeInPlace(t2.TorchTensor) |> ignore

    override _.AddInPlace(t2) = checkMutable(); tt.AddInPlace(t2.TorchTensor) |> ignore

    override _.AddScalarInPlace(t2) = checkMutable(); tt.AddInPlace(t2.TorchTensor) |> ignore

    override _.AddMatrixVecInPlace(t2) = checkMutable(); tt.AddInPlace(t2.TorchTensor) |> ignore

    // TODO - it feels like this should be faster
    override t1.AddSliceInPlace(location, t2) = 
        checkMutable()
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = Shape.unsqueezeAs shape2 shape1
        let t2Expanded = t2.TorchTensor.Expand(toTorchShape expandedShape2)
        let mutable t1Slice = tt // will share memory with res
        for d in 0 .. location.Length - 1 do 
            let len2 = expandedShape2.[d]
            if location.[d] <> 0 || len2 <> shape1.[d] then 
                t1Slice <- t1Slice.Narrow(int64 d, int64 location.[d], int64 len2)
        t1Slice.AddInPlace(t2Expanded) |> ignore

    override _.SubInPlace(t2) = checkMutable(); tt.SubInPlace(t2.TorchTensor) |> ignore

    override _.SubScalarInPlace(t2) = checkMutable(); tt.SubInPlace(t2.TorchTensor) |> ignore

    override _.MulInPlace(t2) = checkMutable(); tt.MulInPlace(t2.TorchTensor) |> ignore

    override _.MulScalarInPlace(t2) = checkMutable(); tt.MulInPlace(t2.TorchTensor) |> ignore

    override _.DivInPlace(t2) = checkMutable(); tt.DivInPlace(t2.TorchTensor) |> ignore

    override _.DivScalarInPlace(t2) = checkMutable(); tt.DivInPlace(t2.TorchTensor) |> ignore

    override _.PowInPlace(t2) = checkMutable(); tt.PowInPlace(t2.TorchTensor) |> ignore

    override _.PowScalarInPlace(t2) = checkMutable(); tt.PowInPlace(t2.TorchTensor) |> ignore

    override _.MatMulInPlace(t2) = checkMutable(); tt <- tt.MatMul(t2.TorchTensor) 

    override _.NegInPlace() = checkMutable(); tt.NegInPlace() |> ignore

    override _.SignInPlace() = checkMutable(); tt.SignInPlace() |> ignore

    override _.FloorInPlace() = checkMutable(); tt.FloorInPlace() |> ignore

    override _.CeilInPlace() = checkMutable(); tt.CeilInPlace() |> ignore

    override _.RoundInPlace() = checkMutable(); tt.RoundInPlace() |> ignore

    override _.AbsInPlace() = checkMutable(); tt.AbsInPlace() |> ignore

    override _.ReluInPlace() = checkMutable(); tt.ReluInPlace() |> ignore

    override _.SoftplusInPlace() = checkMutable(); tt <- tt.Softplus() 

    override _.SigmoidInPlace() = checkMutable(); tt <- tt.Sigmoid() 

    override _.ExpInPlace() = checkMutable(); tt <- tt.Exp()

    override _.LogInPlace() = checkMutable(); tt.LogInPlace() |> ignore

    override _.Log10InPlace() = checkMutable(); tt.Log10InPlace() |> ignore

    override _.SqrtInPlace() = checkMutable(); tt.SqrtInPlace() |> ignore

    override _.SinInPlace() = checkMutable(); tt.SinInPlace() |> ignore

    override _.CosInPlace() = checkMutable(); tt.CosInPlace() |> ignore

    override _.TanInPlace() = checkMutable(); tt.TanInPlace() |> ignore

    override _.SinhInPlace() = checkMutable(); tt.SinhInPlace() |> ignore

    override _.CoshInPlace() = checkMutable(); tt.CosInPlaceh() |> ignore

    override _.TanhInPlace() = checkMutable(); tt.TanhInPlace() |> ignore

    override _.AsinInPlace() = checkMutable(); tt.AsinInPlace() |> ignore

    override _.AcosInPlace() = checkMutable(); tt.AcosInPlace() |> ignore

    override _.AtanInPlace() = checkMutable(); tt.AtanInPlace() |> ignore

    // TODO - next version of TorchSharp will have in place version of this
    override t.OnesInPlace() = checkMutable(); tt <- (RawTensor.Ones(shape, dtype, t.Device, Backend.Torch) :?> TorchRawTensor).TorchTensor

    // TODO - next version of TorchSharp will have in place version of this
    override t.ZerosInPlace() = checkMutable(); tt <- (RawTensor.Zeros(shape, dtype, t.Device, Backend.Torch) :?> TorchRawTensor).TorchTensor

    // TODO - next version of TorchSharp will have in place version of this
    override t.RandomInPlace() = checkMutable(); tt <- (RawTensor.Random(shape, dtype, t.Device, Backend.Torch) :?> TorchRawTensor).TorchTensor

    // TODO - next version of TorchSharp will have in place version of this
    override t.RandomNormalInPlace() = checkMutable(); tt <- (RawTensor.RandomNormal(shape, dtype, t.Device, Backend.Torch) :?> TorchRawTensor).TorchTensor

    // TODO - next version of TorchSharp will have in place version of this
    override t.RandomIntInPlace(low, high) = checkMutable(); tt <- (RawTensor.RandomInt(shape, low, high, dtype, t.Device, Backend.Torch) :?> TorchRawTensor).TorchTensor

    override t.SetImmutable() = isMutable <- false
    override t.SetMutable() = isMutable <- true

    override t.IsMutable = isMutable

/// The parameterized implementation of the static ops. Use a generic class to
/// make sure we get the correlation with .NET types correct and systematic
type TorchTensorOps<'T, 'T2>
       (dtype: Dtype, conv: 'T -> 'T2,
        fromScalar: 'T2 -> TorchTensor,
        from: 'T2[] * TorchShape -> TorchTensor,
        zero: 'T,
        one: 'T,
        empty: TorchShape  * Device -> TorchTensor,
        zeros: TorchShape  * Device -> TorchTensor,
        ones: TorchShape  * Device -> TorchTensor,
        random: TorchShape  * Device -> TorchTensor,
        randomN: TorchShape  * Device -> TorchTensor,
        randomIntegers: TorchShape * int * int * Device -> TorchTensor,
        valueFromObj: obj -> 'T,
        scalarFromConvValue: 'T2 -> TorchScalar) = 

    member _.Zero(device) = TorchRawTensor(torchMoveTo (fromScalar (conv zero)) device, Shape.scalar, dtype, device) :> RawTensor 
    member _.One(device) = TorchRawTensor(torchMoveTo (fromScalar (conv one)) device, Shape.scalar, dtype, device) :> RawTensor
    member _.Empty(shape:Shape, device) = TorchRawTensor(empty(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Zeros(shape:Shape, device) = TorchRawTensor(zeros(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Ones(shape:Shape, device) = TorchRawTensor(ones(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Random(shape:Shape, device) = TorchRawTensor(random(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.RandomNormal(shape:Shape, device) = TorchRawTensor(randomN(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.RandomInt(shape, low, high, device) = TorchRawTensor(randomIntegers(toTorchShape shape, low, high, device), shape, dtype, device) :> RawTensor

    member _.Full(shape:Shape, value:obj, device) =
        let t = zeros(toTorchShape shape, device)
        t.FillInPlace(scalarFromConvValue (conv (valueFromObj value))) |> ignore
        TorchRawTensor(t, shape, dtype, device) :> RawTensor

    member _.CreateFromFlatArray(values:Array, shape:Shape, device:Device) : RawTensor =
        let values = values :?> 'T[] |> Array.map conv 
        let t = 
            match shape with 
            | [| |] -> fromScalar(values.[0])
            | _ -> from (values, toTorchShape shape)
        let tt = torchMoveTo t device
        TorchRawTensor(tt, shape, dtype, device) :> RawTensor

type TorchFloat32TensorOps() = 

    inherit TorchTensorOps<single, single>(Dtype.Float32, id, 
        (fun v -> FloatTensor.From(v)), 
        (fun (data, shape) -> FloatTensor.From(data, shape)), 
        0.0f, 1.0f, 
        (fun (shape, device) -> FloatTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> FloatTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> FloatTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> FloatTensor.Random(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> FloatTensor.RandomN(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, low, high, device) -> FloatTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((float low).ToScalar())), 
        System.Convert.ToSingle, 
        TorchScalar.op_Implicit)

type TorchFloat64TensorOps() = 

    inherit TorchTensorOps<double, double>(Dtype.Float64, id, 
        (fun v -> DoubleTensor.From(v)), 
        (fun (data, shape) -> DoubleTensor.From(data, shape)), 
        0.0, 1.0, 
        (fun (shape, device) -> DoubleTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> DoubleTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> DoubleTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> DoubleTensor.Random(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> DoubleTensor.RandomN(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, low, high, device) -> DoubleTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((double low).ToScalar())), 
        System.Convert.ToDouble, 
        TorchScalar.op_Implicit)

type TorchInt8TensorOps() = 

    inherit TorchTensorOps<sbyte, sbyte>(Dtype.Int8, sbyte,
        (fun v -> SByteTensor.From(v)), 
        (fun (data, shape) -> SByteTensor.From(data, shape)), 
        0y, 1y,
        (fun (shape, device) -> SByteTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> SByteTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> SByteTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Int8), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int8), 
        (fun (shape, low, high, device) -> SByteTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((sbyte low).ToScalar())), 
        System.Convert.ToSByte, 
        TorchScalar.op_Implicit)

type TorchInt16TensorOps() = 

    inherit TorchTensorOps<int16, int16>(Dtype.Int16, int16, 
        (fun v -> ShortTensor.From(v)), 
        (fun (data, shape) -> ShortTensor.From(data, shape)), 
        0s, 1s,
        (fun (shape, device) -> ShortTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> ShortTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> ShortTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Int16), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int16), 
        (fun (shape, low, high, device) -> ShortTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((int16 low).ToScalar())), 
        System.Convert.ToInt16, 
        TorchScalar.op_Implicit)

type TorchInt32TensorOps() = 

    inherit TorchTensorOps<int32, int32>(Dtype.Int32, int32, 
        (fun v -> IntTensor.From(v)), 
        IntTensor.From, 
        0, 1,
        (fun (shape, device) -> IntTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> IntTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> IntTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Int32), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int32), 
        (fun (shape, low, high, device) -> IntTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((int32 low).ToScalar())), 
        System.Convert.ToInt32, 
        TorchScalar.op_Implicit)

type TorchInt64TensorOps() = 

    inherit TorchTensorOps<int64, int64>(Dtype.Int64, int64, 
        (fun v -> LongTensor.From(v)), 
        (fun (data, shape) -> LongTensor.From(data, shape)), 
        0L, 1L,
        (fun (shape, device) -> LongTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> LongTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> LongTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Int64), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int64), 
        (fun (shape, low, high, device) -> LongTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((int64 low).ToScalar())), 
        System.Convert.ToInt64, 
        TorchScalar.op_Implicit)

type TorchBoolTensorOps() = 

    inherit TorchTensorOps<bool, bool>(Dtype.Bool, id, 
        (fun v -> BoolTensor.From(v)), 
        (fun (data, shape) -> BoolTensor.From(data, shape)), 
        false, true,
        (fun (shape, device) -> BoolTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> BoolTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> BoolTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Bool), 
        (fun _ -> opNotSupported "RandomNormal"  Dtype.Bool), 
        (fun (shape, low, high, device) -> BoolTensor.RandomIntegers(min 2L (int64 (high-low)), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((low > 0).ToScalar())), 
        System.Convert.ToBoolean, 
        TorchScalar.op_Implicit)

type TorchByteTensorOps() = 

    inherit TorchTensorOps<byte, byte>(Dtype.Byte, id, 
        (fun v -> ByteTensor.From(v)), 
        (fun (data, shape) -> ByteTensor.From(data, shape)), 
        0uy, 1uy,
        (fun (shape, device) -> ByteTensor.Empty(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> ByteTensor.Zeros(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun (shape, device) -> ByteTensor.Ones(shape, device.TorchDeviceType, device.DeviceIndex)), 
        (fun _ -> opNotSupported "Random" Dtype.Byte), 
        (fun _ -> opNotSupported "RandomNormal"  Dtype.Byte), 
        (fun (shape, low, high, device) -> ByteTensor.RandomIntegers(int64 (high-low), shape, device.TorchDeviceType, device.DeviceIndex).AddInPlace((byte low).ToScalar())), 
        System.Convert.ToByte, 
        TorchScalar.op_Implicit)

type TorchBackendTensorStatics() =
    inherit BackendTensorStatics()

    let torchFloat32 = TorchFloat32TensorOps()
    let torchFloat64 = TorchFloat64TensorOps()
    let torchInt8 = TorchInt8TensorOps()
    let torchInt16 = TorchInt16TensorOps()
    let torchInt32 = TorchInt32TensorOps()
    let torchInt64 = TorchInt64TensorOps()
    let torchByte = TorchByteTensorOps()
    let torchBool = TorchBoolTensorOps()
    let supported = Array.zeroCreate<int> 32
    let isSupported (deviceType: DiffSharp.DeviceType) = 
        let n = int deviceType
        match supported.[n] with 
        | 0 ->
            try
                FloatTensor.Empty([| 1L |], deviceType= enum (int deviceType), deviceIndex=0) |> ignore
                supported.[n] <- 1
                true
             with _ -> 
                supported.[n] <- 2
                false
        | 1 -> true
        | _ -> false

    override _.GetDevices(deviceType) = 
        [ 
          match deviceType with
          | None | Some DiffSharp.DeviceType.CPU ->
              yield Device.CPU
          | _ -> ()

          match deviceType with
          | None | Some DiffSharp.DeviceType.CUDA ->
              if Torch.IsCudaAvailable() then 
                  let ncuda = Torch.CudaDeviceCount()
                  for i in 0 .. ncuda - 1 do
                      yield (Device(DiffSharp.DeviceType.CUDA, i))
          | _ -> ()
          // We don't report other devices in GetDevices as yet though they may be usable
          // There is currently no way in TorchSHarp to get the device count for other device types,
          // you have to work it out via some other route.
        ]

    override _.IsDeviceTypeSupported (deviceType) =
        match deviceType with 
        | DiffSharp.DeviceType.CPU -> true
        | DiffSharp.DeviceType.CUDA -> Torch.IsCudaAvailable()
        | _ -> isSupported deviceType

    override _.Seed(seed) =
        // TODO (important): we need to do *both* this Torch.SetSeed and CUDA SetSeed when device is GPU. CPU seed and CUDA seed are handled separately in torch and libtorch.
        // However at the point of writing this comment, Cuda SetSeed was not available in TorchSharp
        Torch.SetSeed(int64 seed) 

    override _.Zero(dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.Zero(device)
        | Float64 -> torchFloat64.Zero(device)
        | Int8 -> torchInt8.Zero(device)
        | Byte -> torchByte.Zero(device)
        | Int16 -> torchInt16.Zero(device)
        | Int32 -> torchInt32.Zero(device)
        | Int64 -> torchInt64.Zero(device)
        | Bool -> torchBool.Zero(device)

    override _.One(dtype, device) = 
        match dtype with 
        | Float32 -> torchFloat32.One(device)
        | Float64 -> torchFloat64.One(device)
        | Int8 -> torchInt8.One(device)
        | Byte -> torchByte.One(device)
        | Int16 -> torchInt16.One(device)
        | Int32 -> torchInt32.One(device)
        | Int64 -> torchInt64.One(device)
        | Bool -> torchBool.One(device)

    override _.Zeros(shape:Shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.Zeros(shape, device)
        | Float64 -> torchFloat64.Zeros(shape, device)
        | Int8 -> torchInt8.Zeros(shape, device)
        | Byte -> torchByte.Zeros(shape, device)
        | Int16 -> torchInt16.Zeros(shape, device)
        | Int32 -> torchInt32.Zeros(shape, device)
        | Int64 -> torchInt64.Zeros(shape, device)
        | Bool -> torchBool.Zeros(shape, device)

    override _.Empty(shape:Shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.Empty(shape, device)
        | Float64 -> torchFloat64.Empty(shape, device)
        | Int8 -> torchInt8.Empty(shape, device)
        | Byte -> torchByte.Empty(shape, device)
        | Int16 -> torchInt16.Empty(shape, device)
        | Int32 -> torchInt32.Empty(shape, device)
        | Int64 -> torchInt64.Empty(shape, device)
        | Bool -> torchBool.Empty(shape, device)

    override _.Ones(shape:Shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.Ones(shape, device)
        | Float64 -> torchFloat64.Ones(shape, device)
        | Int8 -> torchInt8.Ones(shape, device)
        | Byte -> torchByte.Ones(shape, device)
        | Int16 -> torchInt16.Ones(shape, device)
        | Int32 -> torchInt32.Ones(shape, device)
        | Int64 -> torchInt64.Ones(shape, device)
        | Bool -> torchBool.Ones(shape, device)

    override _.Full(shape:Shape, value:obj, dtype, device) = 
        match dtype with 
        | Float32 -> torchFloat32.Full(shape, value, device)
        | Float64 -> torchFloat64.Full(shape, value, device)
        | Int8 -> torchInt8.Full(shape, value, device)
        | Byte -> torchByte.Full(shape, value, device)
        | Int16 -> torchInt16.Full(shape, value, device)
        | Int32 -> torchInt32.Full(shape, value, device)
        | Int64 -> torchInt64.Full(shape, value, device)
        | Bool -> torchBool.Full(shape, value, device)

    override _.Random(shape:Shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.Random(shape, device)
        | Float64 -> torchFloat64.Random(shape, device)
        | Int8 -> torchInt8.Random(shape, device)
        | Byte -> torchByte.Random(shape, device)
        | Int16 -> torchInt16.Random(shape, device)
        | Int32 -> torchInt32.Random(shape, device)
        | Int64 -> torchInt64.Random(shape, device)
        | Bool -> torchBool.Random(shape, device)

    override _.RandomNormal(shape:Shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.RandomNormal(shape, device)
        | Float64 -> torchFloat64.RandomNormal(shape, device)
        | Int8 -> torchInt8.RandomNormal(shape, device)
        | Byte -> torchByte.RandomNormal(shape, device)
        | Int16 -> torchInt16.RandomNormal(shape, device)
        | Int32 -> torchInt32.RandomNormal(shape, device)
        | Int64 -> torchInt64.RandomNormal(shape, device)
        | Bool -> torchBool.RandomNormal(shape, device)

    override _.RandomInt(shape:Shape, low:int, high:int, dtype, device) = 
        match dtype with 
        | Float32 -> torchFloat32.RandomInt(shape, low, high, device)
        | Float64 -> torchFloat64.RandomInt(shape, low, high, device)
        | Int8 -> torchInt8.RandomInt(shape, low, high, device)
        | Byte -> torchByte.RandomInt(shape, low, high, device)
        | Int16 -> torchInt16.RandomInt(shape, low, high, device)
        | Int32 -> torchInt32.RandomInt(shape, low, high, device)
        | Int64 -> torchInt64.RandomInt(shape, low, high, device)
        | Bool -> torchBool.RandomInt(shape, low, high, device)

    override _.CreateFromFlatArray(values:Array, shape, dtype, device) =
        match dtype with 
        | Float32 -> torchFloat32.CreateFromFlatArray(values, shape, device)
        | Float64 -> torchFloat64.CreateFromFlatArray(values, shape, device)
        | Int8 -> torchInt8.CreateFromFlatArray(values, shape, device)
        | Byte -> torchByte.CreateFromFlatArray(values, shape, device)
        | Int16 -> torchInt16.CreateFromFlatArray(values, shape, device)
        | Int32 -> torchInt32.CreateFromFlatArray(values, shape, device)
        | Int64 -> torchInt64.CreateFromFlatArray(values, shape, device)
        | Bool -> torchBool.CreateFromFlatArray(values, shape, device)

