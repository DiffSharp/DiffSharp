// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open TorchSharp

type torch_cuda = torch.cuda
type TorchShape = int64[]
type TorchDevice = Torch.Device
type Device = DiffSharp.Device
[<AutoOpen>]
module internal Utils = 

    let int64s (b: int[]) = Array.map int64 b

    let toTorchType dtype =
        match dtype with 
        | Dtype.Bool -> torch.ScalarType.Bool
        | Dtype.Int8 -> torch.ScalarType.Int8
        | Dtype.Byte -> torch.ScalarType.Byte
        | Dtype.Int16 -> torch.ScalarType.Int16
        | Dtype.Int32 -> torch.ScalarType.Int32
        | Dtype.Int64 -> torch.ScalarType.Int64
        | Dtype.Float16 -> torch.ScalarType.Float16
        | Dtype.BFloat16 -> torch.ScalarType.BFloat16
        | Dtype.Float32 -> torch.ScalarType.Float32
        | Dtype.Float64 -> torch.ScalarType.Float64

    /// WARNING: TorchSharp Scalar creation is buggy and doesn't preserve types: https://github.com/xamarin/TorchSharp/issues/331
    let toTorchScalar (x: scalar) =
        match x.GetTypeCode() with 
        | TypeCode.Single -> Scalar.op_Implicit (x.toSingle())
        | TypeCode.Double -> Scalar.op_Implicit (x.toDouble())
        | TypeCode.Int32 -> Scalar.op_Implicit (x.toInt32())
        | TypeCode.Int64 -> Scalar.op_Implicit (x.toInt64())
        | TypeCode.Byte -> Scalar.op_Implicit (x.toByte())
        | TypeCode.SByte -> Scalar.op_Implicit (x.toSByte())
        | TypeCode.Int16 -> Scalar.op_Implicit (x.toInt16())
        | TypeCode.Boolean -> Scalar.op_Implicit (x.toBool())
        | t -> failwithf "unknown scalar type '%A'" t

    let fromTorchType ttype =
        match ttype with 
        | torch.ScalarType.Bool -> Dtype.Bool
        | torch.ScalarType.Int8 -> Dtype.Int8
        | torch.ScalarType.Byte -> Dtype.Byte
        | torch.ScalarType.Int16 -> Dtype.Int16
        | torch.ScalarType.Int32 -> Dtype.Int32
        | torch.ScalarType.Int64 -> Dtype.Int64
        | torch.ScalarType.Float32 -> Dtype.Float32
        | torch.ScalarType.Float64 -> Dtype.Float64
        |  _ -> failwith "fromTorchType - other type"

    let toTorchShape (shape: Shape) : TorchShape = int64s shape

    let fromTorchShape (shape: int64[]) = shape |> Array.map int

    type DiffSharp.DeviceType with 
        member x.ToTorch : TorchSharp.DeviceType = enum (int x)

    type DiffSharp.Device with 
        member x.ToTorch = torch.Device(x.DeviceType.ToTorch, x.DeviceIndex)

    let fromTorchDeviceType (x: TorchSharp.DeviceType) : DiffSharp.DeviceType = enum (int x)

    let fromTorchDevice (x: torch.Device) = DiffSharp.Device(fromTorchDeviceType x.``type``, x.index)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    let torchMoveTo (tt: torch.Tensor) (device: Device) =
        tt.``to``(device.ToTorch)

    type RawTensor with
        member x.TorchTensor = (x :?> TorchRawTensor).TorchTensor

/// This is the base class for all RawTensorXyz tuypes.
/// All type-independent operations are implemented directly on this class. 
type TorchRawTensor(tt: torch.Tensor, shape: Shape, dtype: Dtype, device: Device) =

    inherit RawTensor()

    // Note, shape and dtype are stored as fields. These dupicate information in TorchTensor, but
    // it is a little too costly to repeatedly re-extract this information.
    //
    // 'device' is not stored as a field, it is rarely accessed and can be fetched from TorchTensor

#if DEBUG
    // Check the invariants associated with the tensors
    do 
       if tt.dtype <> toTorchType dtype then
           failwithf "mismatched Torch tensor type, expected %A, got %A" (toTorchType dtype) tt.dtype

       if int tt.device_type <> int device.DeviceType then
           failwithf "mismatched Torch tensor device, expected %A, got %A" tt.device_type device.DeviceType

       if int tt.device_index <> int device.DeviceIndex then
           failwithf "mismatched Torch tensor index, expected %A, got %A" tt.device_index device.DeviceIndex

       if toTorchShape shape <> tt.shape then 
           failwithf "mismatched Torch tensor shape, expected %A, got %A" (toTorchShape shape) tt.shape

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
    override _.DeviceType : DiffSharp.DeviceType = enum (int tt.device_type)
    override t.Device = DiffSharp.Device(t.DeviceType, tt.device_index)
    override _.Backend = Backend.Torch
    override _.Handle = box tt

    new (tt: torch.Tensor) =
        TorchRawTensor(tt, fromTorchShape tt.shape, fromTorchType tt.dtype, fromTorchDevice tt.device)

    member t.MakeLike(tt, ?shape, ?dtype, ?device) : RawTensor =
        upcast TorchRawTensor(tt, defaultArg shape t.Shape, defaultArg dtype t.Dtype, defaultArg device t.Device)

    member _.TorchTensor = tt

    override t.GetItem(indexes:int[]) =
        Shape.checkCanIndex t.Shape indexes
        if t.Shape.Length = 0 then t.ToScalar()
        else t.MakeLike(tt=tt[indexes |> Array.map (fun v -> torch.TensorIndex.Single(int64 v))], shape=[||]).ToScalar()

    override t.GetSlice(fullBounds:int[,]) =
        let n = fullBounds.GetLength(0)
        let newShape = Shape.checkCanGetSlice t.Shape fullBounds

        let indices =
            Array.init n (fun i -> 
                let start = fullBounds[i,0]
                let stop = fullBounds[i,1] + 1
                let len = stop - start
                if fullBounds[i,2] = 1 && len = 1 then
                    torch.TensorIndex.Single(int64 start)
                else
                    torch.TensorIndex.Slice(start=int64 start, stop=int64 stop))
        let res = tt.index(indices)
        t.MakeLike(tt=res, shape=newShape)

    override t.Clone() =
        t.MakeLike(tt.clone())

    override t.ComputeHash() = 
        // Torch Tensors must be CPU before Data can be accessed
        let tt = torchMoveTo tt Device.CPU

        let shape = t.Shape
        let mutable res = hash shape
        let n = shapeLength shape
        match dtype with 
        | Dtype.Int8 ->
            let data = tt.data<sbyte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i])
        | Dtype.Byte ->
            let data = tt.data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i])
        | Dtype.Bool ->
            let data = tt.data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i])
        | Dtype.Int16 ->
            let data = tt.data<int16>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i] )
        | Dtype.Int32 ->
            let data = tt.data<int32>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i])
        | Dtype.Int64 -> 
            let data = tt.data<int64>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data[int64 i])
        | Dtype.Float16 ->
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash (tt.ReadCpuFloat16(int64 i)))
        | Dtype.BFloat16 ->
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash (tt.ReadCpuBFloat16(int64 i)))
        | Dtype.Float32 ->
            let data = tt.data<single>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data[int64 i])
        | Dtype.Float64 ->
            let data = tt.data<double>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data[int64 i])
        res
    
    override t.Expand(newShape) =
        t.MakeLike(tt.expand(toTorchShape newShape), shape=newShape)

    override _.ToScalar() : scalar =
        match dtype with 
        | Dtype.Bool -> tt.ToBoolean() :> scalar
        | Dtype.Byte -> tt.ToByte() :> scalar
        | Dtype.Int8 -> tt.ToSByte() :> scalar
        | Dtype.Int16 -> tt.ToInt16() :> scalar
        | Dtype.Int32 -> tt.ToInt32() :> scalar
        | Dtype.Int64 -> tt.ToInt64() :> scalar
        | Dtype.Float16 -> tt.ToSingle() :> scalar
        | Dtype.BFloat16 -> tt.ToSingle() :> scalar
        | Dtype.Float32 -> tt.ToSingle() :> scalar
        | Dtype.Float64 -> tt.ToDouble() :> scalar

    member t.ToValuesTyped<'T>(conv: torch.Tensor -> 'T) : obj =
        // Move the tensors to CPU for efficiency since we're accessing all the data anyway
        let tt = torchMoveTo tt Device.CPU
        match t.Shape with
        | [|  |] -> tt.ToScalar() |> box
        | [| d0 |] -> upcast Array.init<'T> d0 (fun i -> tt[int64 i] |> conv)
        | [| d0; d1 |] -> upcast Array2D.init<'T> d0 d1 (fun i j -> tt[int64 i, int64 j] |> conv)
        | [| d0; d1; d2 |]  -> upcast Array3D.init<'T> d0 d1 d2 (fun i j k -> tt[int64 i, int64 j, int64 k] |> conv)
        | [| d0; d1; d2; d3 |]  -> upcast Array4D.init<'T> d0 d1 d2 d3 (fun i j k l -> tt[int64 i, int64 j, int64 k, int64 l] |> conv)
        | [| d0; d1; d2; d3; d4 |]  -> upcast Array5D.init<'T> d0 d1 d2 d3 d4 (fun i j k l m -> tt[int64 i, int64 j, int64 k, int64 l, int64 m] |> conv)
        | [| d0; d1; d2; d3; d4; d5 |]  -> upcast Array6D.init<'T> d0 d1 d2 d3 d4 d5 (fun i j k l m n -> tt[int64 i, int64 j, int64 k, int64 l, int64 m, int64 n] |> conv)
        | _ -> failwithf "Cannot get array for Tensor dimensions > 6. Consider slicing the Tensor. Shape: %A" t.Shape

    override t.ToValues() =
        match dtype with 
        | Dtype.Bool -> t.ToValuesTyped<bool>(fun s -> s.ToBoolean())
        | Dtype.Byte -> t.ToValuesTyped<byte>(fun s -> s.ToByte())
        | Dtype.Int8 -> t.ToValuesTyped<sbyte>(fun s -> s.ToSByte())
        | Dtype.Int16 -> t.ToValuesTyped<int16>(fun s -> s.ToInt16())
        | Dtype.Int32 -> t.ToValuesTyped<int32>(fun s -> s.ToInt32())
        | Dtype.Int64 -> t.ToValuesTyped<int64>(fun s -> s.ToInt64())
        | Dtype.Float16 -> t.ToValuesTyped<float32>(fun s -> s.ToSingle())
        | Dtype.BFloat16 -> t.ToValuesTyped<float32>(fun s -> s.ToSingle())
        | Dtype.Float32 -> t.ToValuesTyped<float32>(fun s -> s.ToSingle())
        | Dtype.Float64 -> t.ToValuesTyped<double>(fun s -> s.ToDouble())

    member private _.ToRawDataViaDirectAccess< 'T when 'T: struct and 'T :> ValueType and 'T : (new : unit -> 'T) >() : 'T[] =
        // Torch Tensors must be CPU before raw data can be accessed
        let tt2 = torchMoveTo tt Device.CPU

        let data = tt2.data<'T>()
        let res = Array.zeroCreate<'T> (int32 tt2.NumberOfElements)
        for i in 0 .. int32 tt2.NumberOfElements - 1 do
            res[i] <- data[int64 i]
        res

    member t.ToRawData() : Array =
        match dtype with 
        | Dtype.Bool -> t.ToRawDataViaDirectAccess<bool>() :> _
        | Dtype.Byte -> t.ToRawDataViaDirectAccess<byte>() :> _
        | Dtype.Int8 -> t.ToRawDataViaDirectAccess<sbyte>() :> _
        | Dtype.Int16 -> t.ToRawDataViaDirectAccess<int16>() :> _
        | Dtype.Int32 -> t.ToRawDataViaDirectAccess<int32>() :> _
        | Dtype.Int64 -> t.ToRawDataViaDirectAccess<int64>() :> _
        | Dtype.Float32 -> t.ToRawDataViaDirectAccess<float32>() :> _
        | Dtype.Float64 -> t.ToRawDataViaDirectAccess<double>() :> _
        | Dtype.Float16 -> 
            // Move the tensors to CPU for efficiency since we're accessing all the data anyway
            let tt2 = torchMoveTo tt Device.CPU
            Array.init<float32> (int32 tt2.NumberOfElements) (int64 >> tt2.ReadCpuFloat16) :> _
        | Dtype.BFloat16 -> 
            // Move the tensors to CPU for efficiency since we're accessing all the data anyway
            let tt2 = torchMoveTo tt Device.CPU
            Array.init<float32> (int32 tt2.NumberOfElements) (int64 >> tt2.ReadCpuBFloat16) :> _

    override _.StackTs(tensors, dim) =
        let tts, shapes = tensors |> Array.map (fun t -> (t :?> TorchRawTensor).TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _shape2, newShape = Shape.checkCanStack shapes dim
        let result = torch.stack(tts, int64 dim)
        (tensors[0] :?> TorchRawTensor).MakeLike(result, newShape)

    override t.UnstackT(dim) = 
        let shape = t.Shape
        let _shape1, _shape2, unstackedShape = Shape.checkCanUnstack shape dim
        let results = tt.unbind(dim)
        results |> Array.map (fun rvalues -> t.MakeLike(rvalues, shape=unstackedShape))

    override t.CatTs(tensors, dim) = 
        let values, shapes = tensors |> Array.map (fun t -> t.TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _m2, _shape3, outShape = Shape.checkCanCat shapes dim
        let result = torch.cat(values, int64 dim)
        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        let results = tt.split(int64s sizes, int64 dim)
        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, shape=outShape))

    override t.PermuteT(permutation) =
        let _, newShape = Shape.checkCanPermute t.Shape permutation
        let result = tt.permute(int64s permutation)
        t.MakeLike(result, shape=newShape)

    override t.TransposeT(dim0, dim1) =
        Shape.checkCanTranspose t.Shape dim0 dim1
        let result = tt.transpose(int64 dim0, int64 dim1)
        let shape = result.shape |> Array.map int32
        t.MakeLike(result, shape=shape)

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        let newShape = Shape.computeTranspose2d t.Shape
        let result = tt.t()
        t.MakeLike(result, shape=newShape)

    override t.InverseT() =
        Shape.checkCanInvert t.Shape
        let result = tt.inverse()
        t.MakeLike(result, shape=t.Shape)

    override t.DetT() =
        Shape.checkCanDet t.Shape
        let result = torch.linalg.det(tt)
        let shape = result.shape |> Array.map int32
        t.MakeLike(result, shape=shape)

    override t1.SolveTT(t2) =
        let newShape = Shape.checkCanSolve t1.Shape t2.Shape
        let result = torch.linalg.solve(tt, t2.TorchTensor)
        t1.MakeLike(result, shape=newShape)

    override t.SqueezeT(dim) = 
        let shape = t.Shape
        let newShape = Shape.squeeze dim shape
        let mutable res = tt
        let mutable c = 0
        for i in 0 .. t.Dim - 1 do
            if shape[i] = 1 && (dim = -1 || i = dim) then 
                res <- res.squeeze(int64 c)
            else   
                c <- c + 1
        t.MakeLike(res, shape=newShape)

    override t.UnsqueezeT(dim) = 
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        t.MakeLike(tt.unsqueeze(int64 dim), shape=outputShape)

    override t.FlipT(dims:int[]) = 
        // "flip_cuda" not implemented for 'Bool'"
        let result =
            if dtype = Dtype.Bool then 
                tt.to_type(torch.ScalarType.Byte).flip(int64s dims).to_type(torch.ScalarType.Bool)
            elif dtype = Dtype.Float16 || dtype = Dtype.BFloat16  then 
                tt.to_type(torch.ScalarType.Float32).flip(int64s dims).to_type(toTorchType dtype)
            else
                tt.flip(int64s dims)
        t.MakeLike(result)

    override t.DilateT(dilations:int[]) = 
        Shape.checkCanDilate t.Dim dilations
        let outputShape = Shape.dilated t.Shape dilations
        let dims = dilations.Length
        let mutable res = tt
        for i=0 to dims-1 do
            let s = res.shape
            s[i] <- int64 outputShape[i]
            let resnew = t.ZerosLike(fromTorchShape s)
            let indices = Array.init t.Shape[i] id |> Array.map ((*) dilations[i] >> int64)
            let mutable d = TorchInt64TensorOps().CreateFromFlatArray(indices, shape=[|t.Shape[i]|], device=t.Device)
            for _=0 to i-1 do
                d <- d.UnsqueezeT(0)
            for _=i+1 to dims-1 do
                d <- d.UnsqueezeT(d.Dim)
            d <- d.Expand(fromTorchShape res.shape)
            res <- resnew.TorchTensor.scatter(int64 i, d.TorchTensor, res)
        t.MakeLike(res, outputShape)

    override t.UndilateT(dilations:int[]) =
        let shape = t.Shape
        let outputShape = Shape.undilatedShape shape dilations
        let mutable res = tt
        for d in 0 .. dilations.Length - 1 do
            res <- res.slice(int64 d, 0L, int64 shape[d], int64 dilations[d])
        t.MakeLike(res, outputShape)

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype

        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = indices.Cast(Dtype.Int64)
        let res = 
            // LibTorch Gather on float16/bfloat16 gives : method_name not implemented for 'BFloat16'
            if dtype = Dtype.Float16 || dtype = Dtype.BFloat16  then 
                tt.to_type(torch.ScalarType.Float32).gather(int64 dim, indices.TorchTensor).to_type(toTorchType dtype)
            else
                t.TorchTensor.gather(int64 dim, indices.TorchTensor)
        t.MakeLike(res, indices.Shape)

    override t.ScatterT(dim:int, indices, destinationShape:Shape) =
        Shape.checkCanScatter t.Shape dim indices.Shape indices.Dtype destinationShape
        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = indices.Cast(Dtype.Int64)
        let res = t.ZerosLike(destinationShape)
        // LibTorch Scatter on float16/bfloat16 gives : method_name not implemented for 'BFloat16'
        if dtype = Dtype.Float16 || dtype = Dtype.BFloat16  then 
            let res2 = res.TorchTensor.to_type(torch.ScalarType.Float32)
            res2.scatter_(int64 dim, indices.TorchTensor, t.TorchTensor.to_type(torch.ScalarType.Float32)) |> ignore
            t.MakeLike(res2.to_type(toTorchType dtype), destinationShape)
        else
            res.TorchTensor.scatter_(int64 dim, indices.TorchTensor, t.TorchTensor) |> ignore
            res

    override t.ViewT(shape:Shape) =
        Shape.checkCanView t.Shape shape
        t.MakeLike(tt.reshape(toTorchShape shape), shape=shape)  // Use Reshape instead of View to ensure underlying non-contiguous libtorch tensors can be viewed. Internally Reshape uses View if possible, otherwise it copies data to a contiguous tensor and then views.

    override t.Cast(newDtype: Dtype) =
        if newDtype = dtype then 
            upcast t
        else 
            let result = tt.to_type(toTorchType newDtype)
            t.MakeLike(result, dtype=newDtype)

    override t.MoveTo(device) =
        if t.Device = device then (t :> _) else
        let tt2 = torchMoveTo tt device
        t.MakeLike(tt2, device=device)

    override t.Equals(t2:RawTensor) : bool = 
        if dtype = t2.Dtype then
            let r1 = (t.Shape = t2.Shape)
            if not r1 then false else
            let tt2 = t2.TorchTensor
            let r2 = tt.Equals(tt2)
            r2
        else 
            opNotSupported2 "Equals" dtype t2.Dtype

    override t.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        if dtype = t2.Dtype then
            match dtype with 
            | Dtype.IntegralOrBool -> t.Equals(t2)
            | Dtype.Float16 | Dtype.BFloat16 -> 
               // Need because LibTorch 1.7.0 says "isfinite" not implemented for 'BFloat16'
               tt.to_type(torch.ScalarType.Float32).allclose(t2.TorchTensor.to_type(torch.ScalarType.Float32), relativeTolerance, absoluteTolerance)
            | _ -> tt.allclose(t2.TorchTensor, relativeTolerance, absoluteTolerance)
        else 
            opNotSupported2 "Equals" dtype t2.Dtype

    override t.ClampT(low, high) =
        let result = tt.clamp(low.TorchTensor.ToScalar(), high.TorchTensor.ToScalar())
        t.MakeLike(result)

    override t1.LtTT(t2) =
        let result = tt.lt(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.GtTT(t2) =
        let result = tt.gt(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.LeTT(t2) = 
        let result = tt.le(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.GeTT(t2) = 
        let result = tt.ge(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.EqTT(t2) = 
        let result = tt.eq(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t1.NeqTT(t2) = 
        let result = tt.ne(t2.TorchTensor)
        t1.MakeLike(result, dtype=Dtype.Bool)

    override t.MaxReduceT(dim, keepDim) = 
        let (struct (maxValues, indexes)) = tt.max(int64 dim, keepdim=keepDim)
        let newShape = Shape.checkCanMinMaxReduce dim keepDim t.Shape
        let maxValuesResult = t.MakeLike(maxValues, shape=newShape)
        let indexesResult = t.MakeLike(indexes, shape=newShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        maxValuesResult, indexesResult

    override t.MaxIndexT() = 
        // LibTorch 1.7.0: Max on float16/bfloat16 causes grief
        let tt = 
            if dtype = Dtype.Float16 || dtype = Dtype.BFloat16 then 
                tt.to_type(torch.ScalarType.Float32)
            else
                tt
        let res = Array.zeroCreate<int64> t.Dim
        let idxs = Array.zeroCreate t.Dim
        let mutable values = tt
        // repeatedly reduce, tracking the recorded index for the final maximum eventually selected
        for i = t.Dim - 1 downto 0 do 
            let (struct (values2, indexes)) = values.max(int64 i)
            values <- values2
            idxs[i] <- indexes

        for i = 0 to t.Dim - 1 do 
            let idx = idxs[i]

            res[i] <- 
                match i with 
                | 0 -> idx.ToInt64()
                | 1 -> idx[res[0]].ToInt64() 
                | 2 -> idx[res[0], res[1]].ToInt64() 
                | 3 -> idx[res[0], res[1], res[2]].ToInt64() 
                | 4 -> idx[res[0], res[1], res[2], res[3]].ToInt64() 
                | 5 -> idx[res[0], res[1], res[2], res[3], res[4]].ToInt64() 
                | 6 -> idx[res[0], res[1], res[2], res[3], res[4], res[5]].ToInt64() 
                | _ -> failwith "MaxIndexT > 6d nyi for torch"
        res |> Array.map int32

    override t.MinReduceT(dim, keepDim) = 
        let (struct (minValues, indexes)) = tt.min(int64 dim, keepdim=keepDim)
        let newShape = Shape.checkCanMinMaxReduce dim keepDim t.Shape
        let minValuesResult = t.MakeLike(minValues, shape=newShape)
        let indexesResult = t.MakeLike(indexes, shape=newShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        minValuesResult, indexesResult

    override t.MinIndexT() = 
        // LibTorch 1.7.0: Min on float16/bfloat16 causes grief
        let tt = 
            if dtype = Dtype.Float16 || dtype = Dtype.BFloat16 then 
                tt.to_type(torch.ScalarType.Float32)
            else
                tt
        let res = Array.zeroCreate<int64> t.Dim
        let idxs = Array.zeroCreate t.Dim
        let mutable values = tt
        // repeatedly reduce, tracking the recorded index for the final minimum eventually selected
        for i = t.Dim - 1 downto 0 do 
            let (struct (values2, indexes)) = values.min(int64 i)
            values <- values2
            idxs[i] <- indexes

        for i = 0 to t.Dim - 1 do 
            let idx = idxs[i]

            res[i] <- 
                match i with 
                | 0 -> idx.ToInt64()
                | 1 -> idx[res[0]].ToInt64() 
                | 2 -> idx[res[0], res[1]].ToInt64() 
                | 3 -> idx[res[0], res[1], res[2]].ToInt64() 
                | 4 -> idx[res[0], res[1], res[2], res[3]].ToInt64() 
                | 5 -> idx[res[0], res[1], res[2], res[3], res[4]].ToInt64() 
                | 6 -> idx[res[0], res[1], res[2], res[3], res[4], res[5]].ToInt64() 
                | _ -> failwith "MinIndexT > 6d nyi for torch"
        res |> Array.map int32
    
    override t1.AddTT(t2, alpha) =
        let result = 
            match alpha with 
            | Some v -> tt.add(t2.TorchTensor, toTorchScalar v)
            | None -> tt.add(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.AddTT0(t2: scalar, ?alpha: scalar) =
        let result = 
            match alpha with 
            | Some v -> tt.add(toTorchScalar t2, toTorchScalar v)
            | None -> tt.add(toTorchScalar t2)
        t1.MakeLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = Shape.unsqueezeAs shape2 shape1
        let t2Expanded = t2.TorchTensor.expand(toTorchShape expandedShape2)
        let res = tt.clone()
        let mutable t1Slice = res // will share memory with res
        for d in 0 .. location.Length - 1 do 
            let len2 = expandedShape2[d]
            if location[d] <> 0 || len2 <> shape1[d] then 
                t1Slice <- t1Slice.narrow(int64 d, int64 location[d], int64 len2)
        t1Slice.add_(t2Expanded) |> ignore
        t1.MakeLike(res)

    override t1.SubTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "SubT" dtype t2.Dtype
        | _ ->
        let result = tt.sub(t2.TorchTensor)
        t1.MakeLike(result)

    override t2.SubFromT0T(t1:scalar) = t2.SubTT0(t1).NegT()

    override t1.SubTT0(t2: scalar) = 
        //let t2v = t2.TorchTensor.ToScalar()
        let result = tt.sub(toTorchScalar t2)
        t1.MakeLike(result)

    override t1.MulTT(t2) = 
        let result = tt.mul(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.MulTT0(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported "MulTT0" dtype
        | _ ->
        let result = tt.mul(toTorchScalar t2)
        t1.MakeLike(result)

    override t1.DivTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "DivTT" dtype t2.Dtype
        | _ ->
        let result = tt.div(t2.TorchTensor)
        // Torch uses "true division" mirroring Python 3
        // https://www.python.org/dev/peps/pep-0238/
        // https://pytorch.org/docs/stable/generated/torch.div.html
        // also see https://github.com/DiffSharp/DiffSharp/issues/239
        let outtype = Dtype.divisionType t1.Dtype t2.Dtype
        t1.MakeLike(result.to_type(toTorchType outtype), dtype=outtype)

    override t2.DivFromT0T(t1: scalar) =
        match dtype with 
        | Dtype.Bool -> opNotSupported "DivT0T" dtype
        | _ ->
        let t1 = t2.FullLike(Shape.scalar, t1, dtype=t1.dtype)
        let result = t1.TorchTensor.div(t2.TorchTensor)
        // Torch uses "true division" mirroring Python 3
        // https://www.python.org/dev/peps/pep-0238/
        // https://pytorch.org/docs/stable/generated/torch.div.html
        // also see https://github.com/DiffSharp/DiffSharp/issues/239
        let outtype = widenScalarForDivision t2.Dtype t1.Dtype
        t2.MakeLike(result.to_type(toTorchType outtype), dtype=outtype)

    override t1.DivTT0(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported "DivTT0" dtype
        | _ ->
        let t2 = toTorchScalar t2
        // let t2 = t1.FullLike(Shape.scalar, t2, dtype=t1.Dtype)
        let result = tt.div(t2)
        // Torch uses "true division" mirroring Python 3
        // https://www.python.org/dev/peps/pep-0238/
        // https://pytorch.org/docs/stable/generated/torch.div.html
        // also see https://github.com/DiffSharp/DiffSharp/issues/239
        let outtype = widenScalarForDivision t1.Dtype (fromTorchType t2.Type)
        t1.MakeLike(result.to_type(toTorchType outtype), dtype=outtype)

    override t1.PowTT(t2) =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowTT" dtype
        | _ -> 
        let result = tt.pow(t2.TorchTensor)
        t1.MakeLike(result)

    override t2.PowFromT0T(t1:scalar) = 
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowT0T" dtype
        | _ -> 
        let t1 = t2.FullLike(Shape.scalar, t1)
        let result = t1.Expand(t2.Shape).TorchTensor.pow(t2.TorchTensor)
        t2.MakeLike(result)

    override t1.PowTT0(t2:scalar) =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "PowTT0" dtype
        | _ -> 
        let t2v = toTorchScalar t2
        let result = tt.pow(t2v)
        t1.MakeLike(result)

    override t1.MatMulTT(t2) = 
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "MatMulTT" dtype t2.Dtype
        | _ ->  
        let (t1BatchPart, t1MatrixPart), (t2BatchPart, t2MatrixPart) = Shape.checkCanMatmul t1.Shape t2.Shape
        if t1BatchPart <> t2BatchPart then failwithf "Cannot matrix multiply raw tensors with shapes %A, %A - mismatch batching" t1.Shape t2.Shape
        let t1rows = t1MatrixPart[0]
        let t2cols = t2MatrixPart[1]
        let newShape = Array.append t1BatchPart [| t1rows; t2cols |]        
        let result =
            // "addmm for CUDA tensors only supports floating-point types. Try converting the tensors with .float()" | const char *
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                let tt1 = tt.to_type(torch.ScalarType.Float64)
                let tt2 = t2.TorchTensor.to_type(torch.ScalarType.Float64)
                tt1.matmul(tt2).round().to_type(toTorchType dtype) 
            | _ ->
                tt.matmul(t2.TorchTensor)
        t1.MakeLike(result, newShape)

    override t1.BMMTT(t2) =
        match dtype with 
        | Dtype.Bool -> opNotSupported2 "BMMTT" dtype t2.Dtype
        | _ ->  
        let resultShape = Shape.checkCanBMM t1.Shape t2.Shape
        let result =
            // "addmm for CUDA tensors only supports floating-point types. Try converting the tensors with .float()" | const char *
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                let tt1 = tt.to_type(torch.ScalarType.Float64)
                let tt2 = t2.TorchTensor.to_type(torch.ScalarType.Float64)
                tt1.bmm(tt2).round().to_type(toTorchType dtype) 
            | _ ->
                tt.bmm(t2.TorchTensor)
        t1.MakeLike(result, resultShape)        

    override t1.Conv1D(t2, stride, padding) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelSize, _outputChannels, _outputSize, outputShape =
            Shape.checkCanConv1d t1.DeviceType t2.DeviceType dtype t2.Dtype t1.Shape t2.Shape stride padding 1
        let resultt =
            // "conv1d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                torch.nn.functional.conv1d(tt.to_type(torch.ScalarType.Float64), t2.TorchTensor.to_type(torch.ScalarType.Float64), stride=int64 stride, padding=int64 padding, dilation=1L).round().to_type(toTorchType dtype) 
            | _ ->
                torch.nn.functional.conv1d(tt, t2.TorchTensor, stride=int64 stride, padding=int64 padding, dilation=1L)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv2D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanConv2d t1.DeviceType t2.DeviceType dtype t2.Dtype t1.Shape t2.Shape strides paddings [| 1;1 |]
        let resultt =
            // "conv2d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                torch.nn.functional.conv2d(tt.to_type(torch.ScalarType.Float64), t2.TorchTensor.to_type(torch.ScalarType.Float64), strides=int64s strides, padding=int64s paddings).round().to_type(toTorchType dtype) 
            | _ ->
                torch.nn.functional.conv2d(tt, t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv3D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanConv3d t1.DeviceType t2.DeviceType dtype t2.Dtype  t1.Shape t2.Shape strides paddings [| 1;1;1 |]
        let resultt =
            // "conv2d for CUDA tensors only supports floating-point types."
            match t1.DeviceType, dtype with 
            | DiffSharp.DeviceType.CUDA, (Dtype.Integral as dtype) ->
                torch.nn.functional.conv3d(tt.to_type(torch.ScalarType.Float64), t2.TorchTensor.to_type(torch.ScalarType.Float64), strides=int64s strides, padding=int64s paddings).round().to_type(toTorchType dtype) 
            | _ ->
                torch.nn.functional.conv3d(tt, t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxPool1D(kernelSize, stride, padding) = 
        let _batchSize, _channels, _inputSize, _outputSize, outputShape =
            Shape.checkCanMaxpool1d dtype t1.Shape kernelSize stride padding
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "MaxPool1D" dtype
        | _ ->
        let struct (resultt, indicest) = torch.nn.functional.max_pool1d_with_indices(tt, int64 kernelSize, stride=int64 stride, padding=int64 padding, dilation=1L)
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool2D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanMaxpool2d dtype t1.Shape kernelSize strides paddings
        let struct (resultt, indicest) = torch.nn.functional.max_pool2d_with_indices(tt, int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=Dtype.Int64).Cast(Dtype.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool3D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape =
            Shape.checkCanMaxpool3d dtype t1.Shape kernelSize strides paddings
        let struct (resultt, indicest) = torch.nn.functional.max_pool3d_with_indices(tt, int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        
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
        let resulttX = t1X.MaxUnpool2D(indicesX, [| outputSize[0]; outputSize[1]; 1; outputSize[2] |])
        let resultt = resulttX.SqueezeT(2)
        resultt

    override t1.MaxUnpool2D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape =
            Shape.checkCanMaxunpool2d dtype t1.Shape indices.Dtype indices.Shape outputSize
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(Dtype.Int64)

        // note, LibTorch only wants the last two elements of the output size passsed in
        // "There should be exactly two elements (height, width) in output_size (max_unpooling2d_shape_check at ...)"
        let outputSize = outputSize[2..3]
        
        // TODO: consider switching to the torch::nn module for MaxUnpool2d

        let resultt = torch.nn.functional.max_unpool2d(tt, indices.TorchTensor, int64s outputSize)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxUnpool3D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape =
            Shape.checkCanMaxunpool3d dtype t1.Shape indices.Dtype indices.Shape outputSize
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(Dtype.Int64)

        // note, LibTorch only wants the last three elements of the output size passsed in
        // "There should be exactly three elements (depth, height, width) in output_size (max_unpooling3d_shape_check at ..\..\aten\src\ATen\native\MaxUnpooling.cpp:231)"
        let outputSize = outputSize[2..4]
        
        // NOTE: strides and padding must always be specified for torch::max_unpool3d C++ entry
        // TODO: consider switching to the torch::nn module for MaxUnpool
        let strides = outputSize |> Array.map (fun _ -> 1L)
        let padding = outputSize |> Array.map (fun _ -> 0L)
        let resultt = torch.nn.functional.max_unpool3d(tt, indices.TorchTensor, int64s outputSize, strides, padding)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.AvgPool1D(kernelSize, stride, padding) =
        let _batchSize, _channels, _inputSize, _outputSize, outputShape = Shape.checkCanAvgpool1d dtype t1.Shape kernelSize stride padding
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "AvgPool1D" dtype
        | _ ->
        let resultt = torch.nn.functional.avg_pool1d(tt, int64 kernelSize, stride=int64 stride, padding=int64 padding)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result

    override t1.AvgPool2D(kernelSize, stride, padding) = 
        let _batchSize, _channels, _inputSize, _kernelSize, _outputSize, outputShape = Shape.checkCanAvgpool2d dtype t1.Shape kernelSize stride padding
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "AvgPool2D" dtype
        | _ ->
        let resultt = torch.nn.functional.avg_pool2d(tt, int64s kernelSize, strides=int64s stride, paddings=int64s padding)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result

    override t1.AvgPool3D(kernelSize, stride, padding) =
        let _batchSize, _channels, _inputSize, _kernelSize, _outputSize, outputShape = Shape.checkCanAvgpool3d dtype t1.Shape kernelSize stride padding
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "AvgPool3D" dtype
        | _ ->
        let resultt = torch.nn.functional.avg_pool3d(tt, int64s kernelSize, strides=int64s stride, paddings=int64s padding)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result

    override t1.AvgPoolReverse1D(originalInput, kernelSize, stride, padding) =
        let t1X = t1.UnsqueezeT(2)
        let originalInputX = originalInput.UnsqueezeT(2)
        let resulttX = t1X.AvgPoolReverse2D(originalInputX, [| 1; kernelSize |], [| 1; stride |], [| 0; padding |])
        let resultt = resulttX.SqueezeT(2)
        resultt

    override t1.AvgPoolReverse2D(originalInput, kernelSize, stride, padding) = 
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "AvgPoolReverse2D" dtype
        | _ ->
        let resultt = torch.nn.functional.avg_pool2d_backward(tt, originalInput.TorchTensor, int64s kernelSize, strides=int64s stride, paddings=int64s padding)
        let result = t1.MakeLike(resultt, shape=originalInput.Shape)
        result

    override t1.AvgPoolReverse3D(originalInput, kernelSize, stride, padding) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "AvgPoolReverse3D" dtype
        | _ ->
        let resultt = torch.nn.functional.avg_pool3d_backward(tt, originalInput.TorchTensor, int64s kernelSize, strides=int64s stride, paddings=int64s padding)
        let result = t1.MakeLike(resultt, shape=originalInput.Shape)
        result

    override t.NegT() =
        match dtype with 
        | Dtype.Bool -> opNotSupported "NegT" dtype
        | _ ->  t.MakeLike(-tt)

    override t.SumT(?resultType) =
        let typeArg = match resultType with None -> Nullable() | Some dt -> Nullable(toTorchType dt)
        let outType = match resultType with None -> dtype.SummationType | Some dt -> dt
        t.MakeLike(tt.sum(typeArg), shape=Shape.scalar, dtype=outType)

    override t.SumTDim(dim, ?resultType) =
        let typeArg = match resultType with None -> Nullable() | Some dt -> Nullable(toTorchType dt)
        let outType = match resultType with None -> dtype.SummationType | Some dt -> dt
        let ret = tt.sum(dim=(int64 dim), ``type``=typeArg, keepdim=false)  // keepdim is fixed to false as it is handled at Tensor level, not at RawTensor level
        t.MakeLike(ret, shape=fromTorchShape ret.shape, dtype=outType)

    override t.SignT() =
        t.MakeLike(tt.sign())

    override t.FloorT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "FloorT" dtype
        | _ ->  t.MakeLike(tt.floor())

    override t.CeilT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CeilT" dtype
        | _ ->  t.MakeLike(tt.ceil())

    override t.RoundT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "RoundT" dtype
        | _ ->  t.MakeLike(tt.round())

    override t.AbsT() = 
        match dtype with 
        | Dtype.Bool -> opNotSupported "AbsT" dtype
        | Dtype.Int8 -> t.Cast(Dtype.Int32).AbsT().Cast(Dtype.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ -> t.MakeLike(tt.abs ())

    override t.SoftplusT() = 
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SoftplusT" dtype
        | _ -> t.MakeLike(tt.softplus())

    override t.ReluT() =
        match dtype with 
        | Dtype.Bool -> opNotSupported "ReluT" dtype
        | Dtype.Int8 -> t.Cast(Dtype.Int32).ReluT().Cast(Dtype.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ ->   t.MakeLike(tt.relu())

    override t.SigmoidT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SigmoidT" dtype
        | _ ->  t.MakeLike(tt.sigmoid())

    override t.ExpT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "ExpT" dtype
        | _ ->  t.MakeLike(tt.exp())

    override t.LogT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "LogT" dtype
        | _ ->  t.MakeLike(tt.log())

    override t.Log10T() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "Log10T" dtype
        | _ ->   t.MakeLike(tt.log10())

    override t.SqrtT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SqrtT" dtype
        | _ ->  t.MakeLike(tt.sqrt())

    override t.SinT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SinT" dtype
        | _ ->  t.MakeLike(tt.sin())

    override t.CosT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CosT" dtype
        | _ ->  t.MakeLike(tt.cos())

    override t.TanT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "TanT" dtype
        | _ ->  t.MakeLike(tt.tan())

    override t.SinhT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "SinhT" dtype
        | _ ->  t.MakeLike(tt.sinh())

    override t.CoshT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "CoshT" dtype
        | _ ->  t.MakeLike(tt.cosh())

    override t.TanhT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "TanhT" dtype
        | _ ->  t.MakeLike(tt.tanh())

    override t.AsinT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AsinT" dtype
        | _ ->  t.MakeLike(tt.asin())

    override t.AcosT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AcosT" dtype
        | _ ->  t.MakeLike(tt.acos())

    override t.AtanT() =
        match dtype with 
        | Dtype.IntegralOrBool -> opNotSupported "AtanT" dtype
        | _ ->  t.MakeLike(tt.atan())
#if LATEST_TORCHSHARP
    // Included to track new functionality available in TorchSharp
    //
    // These will be progressed to RawTensor and Tensor
    member t.AdaptiveAvgPool1D(outputSize: int32) =
        match dtype with 
        | Dtype.Bool -> opNotSupported "AdaptiveAvgPool1D" dtype
        | _ ->  t.MakeLike(tt.AdaptiveAvgPool1D(int64 outputSize))

    member t.AdaptiveAvgPool2D(outputSizes: int32[]) =
        match dtype with 
        | Dtype.Bool -> opNotSupported "AdaptiveAvgPool2D" dtype
        | _ ->  t.MakeLike(tt.AdaptiveAvgPool2D(int64s outputSizes))

    member t.AdaptiveAvgPool3D(outputSizes: int32[]) =
        match dtype with 
        | Dtype.Bool -> opNotSupported "AdaptiveAvgPool3D" dtype
        | _ ->  t.MakeLike(tt.AdaptiveAvgPool3D(int64s outputSizes))

    member t.AdaptiveAvgPool3DBackward(originalInput: RawTensor) =
        match dtype with 
        | Dtype.Bool -> opNotSupported "AdaptiveAvgPool3DBackward" dtype
        | _ ->  t.MakeLike(tt.AdaptiveAvgPool3Backward(originalInput.TorchTensor))

    //member t.AvgPool1D(kernelSize: int32, stride: int32, padding: int32, ?ceil_mode: bool, ?count_include_pad: bool) =
    //    //let _batchSize, _channels, _inputSize, _outputSize, outputShape = Shape.checkCanAvgPool1d dtype t1.Shape kernelSize stride padding
    //    match dtype with 
    //    | Dtype.Bool -> opNotSupported "AvgPool1D" dtype
    //    | _ ->
    //    let _resultt = tt.AvgPool1D(int64 kernelSize, stride=int64 stride, padding=int64 padding, ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad)
    //    failwith "tbd - outputShape"
    //    //t.MakeLike(resultt, shape=outputShape)

    //member t.AvgPool2D(kernelSizes: int32[], strides: int32[], paddings: int32[], ?ceil_mode: bool, ?count_include_pad: bool) =
    //    failwith "tbd - TorchSharp signture being updated"
        ////let _batchSize, _channels, _inputSize, _outputSize, outputShape = Shape.checkCanAvgPool1d dtype t1.Shape kernelSize stride padding
        //match dtype with 
        //| Dtype.Bool -> opNotSupported "AvgPool2D" dtype
        //| _ ->
        //let _resultt = tt.AvgPool2D(int64s kernelSizes, stride=int64 stride, padding=int64 padding, ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad)
        //failwith "tbd - outputShape"
        ////t.MakeLike(resultt, shape=outputShape)

    //member t.X(kernelSize: int32, stride: int32, padding: int32, ?ceil_mode: bool, ?count_include_pad: bool) =
    //    //let _batchSize, _channels, _inputSize, _outputSize, outputShape = Shape.checkCanAvgPool1d dtype t1.Shape kernelSize stride padding
    //    match dtype with 
    //    | Dtype.Bool -> opNotSupported "AvgPool1D" dtype
    //    | _ ->
    //    let _resultt = tt.BitwiseAnd(int64 kernelSize, stride=int64 stride, padding=int64 padding, ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad)
    //    failwith "tbd - outputShape"
    //    //t.MakeLike(resultt, shape=outputShape)
#endif

    new (info: System.Runtime.Serialization.SerializationInfo, _context: System.Runtime.Serialization.StreamingContext) =
        let dtype = info.GetValue("dtype", typeof<Dtype>) :?> Dtype
        let shape = info.GetValue("shape", typeof<Shape>) :?> Shape
        let tt =
            match dtype with 
            | Dtype.Bool -> 
                let data = info.GetValue("data", typeof<bool[]>)  :?> bool[]
                torch.tensor(data, dtype=toTorchType Dtype.Bool, dimensions=toTorchShape shape) 
            | Dtype.Byte -> 
                let data = info.GetValue("data", typeof<byte[]>)  :?> byte[]
                torch.tensor(data, dtype=toTorchType Dtype.Byte, dimensions=toTorchShape shape) 
            | Dtype.Int8 -> 
                let data = info.GetValue("data", typeof<sbyte[]>)  :?> sbyte[]
                torch.tensor(data, dtype=toTorchType Dtype.Int8, dimensions=toTorchShape shape) 
            | Dtype.Int16 -> 
                let data = info.GetValue("data", typeof<int16[]>)  :?> int16[]
                torch.tensor(data, dtype=toTorchType Dtype.Int16, dimensions=toTorchShape shape) 
            | Dtype.Int32 -> 
                let data = info.GetValue("data", typeof<int32[]>)  :?> int32[]
                torch.tensor(data, dtype=toTorchType Dtype.Int32, dimensions=toTorchShape shape) 
            | Dtype.Int64 -> 
                let data = info.GetValue("data", typeof<int64[]>)  :?> int64[]
                torch.tensor(data, dtype=toTorchType Dtype.Int64, dimensions=toTorchShape shape) 
            | Dtype.Float32 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                torch.tensor(data, dtype=toTorchType Dtype.Float32, dimensions=toTorchShape shape) 
            | Dtype.Float64 -> 
                let data = info.GetValue("data", typeof<double[]>)  :?> double[]
                torch.tensor(data, dtype=toTorchType Dtype.Float64, dimensions=toTorchShape shape) 
            | Dtype.Float16 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                torch.tensor(data, dtype=toTorchType Dtype.Float16, dimensions=toTorchShape shape) 
            | Dtype.BFloat16 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                torch.tensor(data, dtype=toTorchType Dtype.BFloat16, dimensions=toTorchShape shape) 

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
        tt <- tt.clamp(low.TorchTensor.ToScalar(), high.TorchTensor.ToScalar())

    override _.LtInPlace(t2) = checkMutable(); tt.lt_(t2.TorchTensor) |> ignore

    override _.GtInPlace(t2) = checkMutable(); tt.gt_(t2.TorchTensor) |> ignore

    override _.LeInPlace(t2) = checkMutable(); tt.le_(t2.TorchTensor) |> ignore

    override _.GeInPlace(t2) = checkMutable(); tt.ge_(t2.TorchTensor) |> ignore

    override _.EqInPlace(t2) = checkMutable(); tt.eq_(t2.TorchTensor) |> ignore

    override _.NeqInPlace(t2) = checkMutable(); tt.ne_(t2.TorchTensor) |> ignore

    override _.AddInPlace(t2, alpha) =
        checkMutable()
        match alpha with 
        | Some v -> tt.add_(t2.TorchTensor, toTorchScalar v) |> ignore
        | None -> tt.add_(t2.TorchTensor) |> ignore

    override _.AddScalarInPlace(t2) =
        checkMutable()
        tt.add_(toTorchScalar t2) |> ignore

    // TODO - this should be faster
    override t1.AddSliceInPlace(location, t2) = 
        checkMutable()
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = Shape.unsqueezeAs shape2 shape1
        let t2Expanded = t2.TorchTensor.expand(toTorchShape expandedShape2)
        let mutable t1Slice = tt // will share memory with res
        for d in 0 .. location.Length - 1 do 
            let len2 = expandedShape2[d]
            if location[d] <> 0 || len2 <> shape1[d] then 
                t1Slice <- t1Slice.narrow(int64 d, int64 location[d], int64 len2)
        t1Slice.add_(t2Expanded) |> ignore

    override _.SubInPlace(t2) = checkMutable(); tt.sub_(t2.TorchTensor) |> ignore

    override _.SubScalarInPlace(t2) = checkMutable(); tt.sub_(toTorchScalar t2) |> ignore

    override _.MulInPlace(t2) = checkMutable(); tt.mul_(t2.TorchTensor) |> ignore

    override _.MulScalarInPlace(t2) = checkMutable(); tt.mul_(toTorchScalar t2) |> ignore

    override _.DivInPlace(t2) = checkMutable(); tt.div_(t2.TorchTensor) |> ignore

    override _.DivScalarInPlace(t2) = checkMutable(); tt.div_(toTorchScalar t2) |> ignore

    override _.PowInPlace(t2) = checkMutable(); tt.pow_(t2.TorchTensor) |> ignore

    override _.PowScalarInPlace(t2) = checkMutable(); tt.pow_(toTorchScalar t2) |> ignore

    override _.MatMulInPlace(t2) = checkMutable(); tt <- tt.matmul(t2.TorchTensor) 

    override _.NegInPlace() = checkMutable(); tt.neg_() |> ignore

    override _.SignInPlace() = checkMutable(); tt.sign_() |> ignore

    override _.FloorInPlace() = checkMutable(); tt.floor_() |> ignore

    override _.CeilInPlace() = checkMutable(); tt.ceil_() |> ignore

    override _.RoundInPlace() = checkMutable(); tt.round_() |> ignore

    override _.AbsInPlace() = checkMutable(); tt.abs_() |> ignore

    override _.ReluInPlace() = checkMutable(); tt.relu_() |> ignore

    override _.SoftplusInPlace() = checkMutable(); tt <- tt.softplus() 

    override _.SigmoidInPlace() = checkMutable(); tt <- tt.sigmoid() 

    override _.ExpInPlace() = checkMutable(); tt <- tt.exp()

    override _.LogInPlace() = checkMutable(); tt.log_() |> ignore

    override _.Log10InPlace() = checkMutable(); tt.log10_() |> ignore

    override _.SqrtInPlace() = checkMutable(); tt.sqrt_() |> ignore

    override _.SinInPlace() = checkMutable(); tt.sin_() |> ignore

    override _.CosInPlace() = checkMutable(); tt.cos_() |> ignore

    override _.TanInPlace() = checkMutable(); tt.tan_() |> ignore

    override _.SinhInPlace() = checkMutable(); tt.sinh_() |> ignore

    override _.CoshInPlace() = checkMutable(); tt.cosh_() |> ignore

    override _.TanhInPlace() = checkMutable(); tt.tanh_() |> ignore

    override _.AsinInPlace() = checkMutable(); tt.asin_() |> ignore

    override _.AcosInPlace() = checkMutable(); tt.acos_() |> ignore

    override _.AtanInPlace() = checkMutable(); tt.atan_() |> ignore

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

    override t.SetMutable() = isMutable <- true

    override t.IsMutable = isMutable

/// The parameterized implementation of the static ops. Use a generic class to
/// make sure we get the correlation with .NET types correct and systematic
type TorchTensorOps<'T, 'T2>
       (dtype: Dtype, conv: 'T -> 'T2,
        fromScalar: 'T2 -> torch.Tensor,
        from: 'T2[] * TorchShape -> torch.Tensor,
        zero: 'T,
        one: 'T,
        empty: TorchShape  * Device -> torch.Tensor,
        zeros: TorchShape  * Device -> torch.Tensor,
        ones: TorchShape  * Device -> torch.Tensor,
        random: TorchShape  * Device -> torch.Tensor,
        randomN: TorchShape  * Device -> torch.Tensor,
        randomIntegers: TorchShape * int * int * Device -> torch.Tensor,
        valueFromScalar: scalar -> 'T,
        scalarFromConvValue: 'T2 -> TorchSharp.Scalar) = 

    member _.Zero(device) = TorchRawTensor(torchMoveTo (fromScalar (conv zero)) device, Shape.scalar, dtype, device) :> RawTensor 
    member _.One(device) = TorchRawTensor(torchMoveTo (fromScalar (conv one)) device, Shape.scalar, dtype, device) :> RawTensor
    member _.Empty(shape:Shape, device) = TorchRawTensor(empty(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Zeros(shape:Shape, device) = TorchRawTensor(zeros(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Ones(shape:Shape, device) = TorchRawTensor(ones(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.Random(shape:Shape, device) = TorchRawTensor(random(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.RandomNormal(shape:Shape, device) = TorchRawTensor(randomN(toTorchShape shape, device), shape, dtype, device) :> RawTensor
    member _.RandomInt(shape, low, high, device) = TorchRawTensor(randomIntegers(toTorchShape shape, low, high, device), shape, dtype, device) :> RawTensor

    member _.Full(shape:Shape, value:scalar, device) =
        let t = empty(toTorchShape shape, device)
        t.fill_(scalarFromConvValue (conv (valueFromScalar value))) |> ignore
        TorchRawTensor(t, shape, dtype, device) :> RawTensor

    member _.CreateFromFlatArray(values:Array, shape:Shape, device: Device) : RawTensor =
        let values = values :?> 'T[] |> Array.map conv 
        // torch.InitializeDevice(device.ToTorch) |> ignore
        let t = 
            match shape with 
            | [| |] -> fromScalar(values[0])
            | _ -> from (values, toTorchShape shape)
        let tt = torchMoveTo t device
        TorchRawTensor(tt, shape, dtype, device) :> RawTensor

type TorchFloat32TensorOps() = 

    inherit TorchTensorOps<single, single>(Dtype.Float32, id, 
        (fun v -> torch.tensor(float v, dtype=toTorchType Dtype.Float32)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Float32)), 
        0.0f, 1.0f, 
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.rand(size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.randn(size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch)), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Float32, device=device.ToTorch).add_((float low).ToScalar())), 
        System.Convert.ToSingle, 
        TorchSharp.Scalar.op_Implicit)

type TorchFloat64TensorOps() = 

    inherit TorchTensorOps<double, double>(Dtype.Float64, id, 
        (fun v -> torch.tensor(v, dtype=toTorchType Dtype.Float64)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Float64)), 
        0.0, 1.0, 
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.rand(size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.randn(size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch)), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Float64, device=device.ToTorch).add_((double low).ToScalar())), 
        System.Convert.ToDouble, 
        TorchSharp.Scalar.op_Implicit)

type TorchInt8TensorOps() = 

    inherit TorchTensorOps<sbyte, sbyte>(Dtype.Int8, sbyte,
        (fun v -> torch.tensor(int64 v, dtype=toTorchType Dtype.Int8)),
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Int8)), 
        0y, 1y,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Int8, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Int8, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Int8, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Int8), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int8), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Int8, device=device.ToTorch).add_((sbyte low).ToScalar())), 
        System.Convert.ToSByte, 
        TorchSharp.Scalar.op_Implicit)

type TorchInt16TensorOps() = 

    inherit TorchTensorOps<int16, int16>(Dtype.Int16, int16, 
        (fun v -> torch.tensor(int64 v, dtype=toTorchType Dtype.Int16)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Int16)), 
        0s, 1s,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Int16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Int16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Int16, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Int16), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int16), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Int16, device=device.ToTorch).add_((int16 low).ToScalar())), 
        System.Convert.ToInt16, 
        TorchSharp.Scalar.op_Implicit)

type TorchInt32TensorOps() = 

    inherit TorchTensorOps<int32, int32>(Dtype.Int32, int32, 
        (fun v -> torch.tensor(int64 v, dtype=toTorchType Dtype.Int32)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Int32)), 
        0, 1,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Int32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Int32, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Int32, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Int32), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int32), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Int32, device=device.ToTorch).add_((int32 low).ToScalar())), 
        System.Convert.ToInt32, 
        TorchSharp.Scalar.op_Implicit)

type TorchInt64TensorOps() = 

    inherit TorchTensorOps<int64, int64>(Dtype.Int64, int64, 
        (fun v -> torch.tensor(v, dtype=toTorchType Dtype.Int64)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Int64)), 
        0L, 1L,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Int64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Int64, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Int64, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Int64), 
        (fun _ -> opNotSupported "RandomNormal" Dtype.Int64), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Int64, device=device.ToTorch).add_((int64 low).ToScalar())), 
        System.Convert.ToInt64, 
        TorchSharp.Scalar.op_Implicit)

type TorchBoolTensorOps() = 

    inherit TorchTensorOps<bool, bool>(Dtype.Bool, id, 
        (fun v -> torch.tensor(v, dtype=toTorchType Dtype.Bool)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Bool)), 
        false, true,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Bool, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Bool, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Bool, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Bool), 
        (fun _ -> opNotSupported "RandomNormal"  Dtype.Bool), 
        (fun (shape, low, high, device) -> torch.randint(min 2L (int64 (high-low)), size=shape, dtype=toTorchType Dtype.Bool, device=device.ToTorch).add_((low > 0).ToScalar())), 
        System.Convert.ToBoolean, 
        TorchSharp.Scalar.op_Implicit)

type TorchByteTensorOps() = 

    inherit TorchTensorOps<byte, byte>(Dtype.Byte, id, 
        (fun v -> torch.tensor(int64 v, dtype=toTorchType Dtype.Byte)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Byte)), 
        0uy, 1uy,
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Byte, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Byte, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Byte, device=device.ToTorch)), 
        (fun _ -> opNotSupported "Random" Dtype.Byte), 
        (fun _ -> opNotSupported "RandomNormal"  Dtype.Byte), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Byte, device=device.ToTorch).add_((byte low).ToScalar())), 
        System.Convert.ToByte, 
        TorchSharp.Scalar.op_Implicit)

type TorchFloat16TensorOps() = 

    inherit TorchTensorOps<single, single>(Dtype.Float16, id, 
        (fun v -> torch.tensor(float v, dtype=toTorchType Dtype.Float16)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.Float16)), 
        0.0f, 1.0f, 
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.rand(size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.randn(size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch)), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.Float16, device=device.ToTorch).add_((float low).ToScalar())), 
        System.Convert.ToSingle, 
        TorchSharp.Scalar.op_Implicit)


type TorchBFloat16TensorOps() = 

    inherit TorchTensorOps<single, single>(Dtype.BFloat16, id, 
        (fun v -> torch.tensor(float v, dtype=toTorchType Dtype.BFloat16)), 
        (fun (data, shape) -> torch.tensor(data, shape, dtype=toTorchType Dtype.BFloat16)), 
        0.0f, 1.0f, 
        (fun (shape, device) -> torch.empty(size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.zeros(size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.ones(size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.rand(size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch)), 
        (fun (shape, device) -> torch.randn(size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch)), 
        (fun (shape, low, high, device) -> torch.randint(int64 (high-low), size=shape, dtype=toTorchType Dtype.BFloat16, device=device.ToTorch).add_((float low).ToScalar())), 
        System.Convert.ToSingle, 
        TorchSharp.Scalar.op_Implicit)

type TorchBackendTensorStatics() =
    inherit BackendTensorStatics()

    let torchFloat16 = TorchFloat16TensorOps()
    let torchBFloat16 = TorchBFloat16TensorOps()
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
        match supported[n] with 
        | 0 ->
            try
                torch.empty([| 1L |], device= torch.Device(deviceType.ToTorch, index=0)) |> ignore
                supported[n] <- 1
                true
             with _ -> 
                supported[n] <- 2
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
              if torch_cuda.is_available() then 
                  let ncuda = torch_cuda.device_count()
                  for i in 0 .. ncuda - 1 do
                      yield (DiffSharp.Device(DiffSharp.DeviceType.CUDA, i))
          | _ -> ()
          // We don't report other devices in GetDevices as yet though they may be usable
          // There is currently no way in TorchSHarp to get the device count for other device types,
          // you have to work it out via some other route.
        ]

    override _.IsDeviceTypeAvailable (deviceType) =
        match deviceType with 
        | DiffSharp.DeviceType.CPU -> true
        | DiffSharp.DeviceType.CUDA -> torch_cuda.is_available()
        | _ -> isSupported deviceType

    override _.Seed(seed) =
        if torch_cuda.is_available() then
            torch_cuda.manual_seed(int64 seed)  |> ignore
        torch.random.manual_seed(int64 seed)  |> ignore

    override _.Zero(dtype, device) =
        match dtype with 
        | Float16 -> torchFloat16.Zero(device)
        | BFloat16 -> torchBFloat16.Zero(device)
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
        | Float16 -> torchFloat16.One(device)
        | BFloat16 -> torchBFloat16.One(device)
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
        | Float16 -> torchFloat16.Zeros(shape, device)
        | BFloat16 -> torchBFloat16.Zeros(shape, device)
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
        | Float16 -> torchFloat16.Empty(shape, device)
        | BFloat16 -> torchBFloat16.Empty(shape, device)
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
        | Float16 -> torchFloat16.Ones(shape, device)
        | BFloat16 -> torchBFloat16.Ones(shape, device)
        | Float32 -> torchFloat32.Ones(shape, device)
        | Float64 -> torchFloat64.Ones(shape, device)
        | Int8 -> torchInt8.Ones(shape, device)
        | Byte -> torchByte.Ones(shape, device)
        | Int16 -> torchInt16.Ones(shape, device)
        | Int32 -> torchInt32.Ones(shape, device)
        | Int64 -> torchInt64.Ones(shape, device)
        | Bool -> torchBool.Ones(shape, device)

    override _.Full(shape:Shape, value:scalar, dtype, device) =
        match dtype with 
        | Float16 -> torchFloat16.Full(shape, value, device)
        | BFloat16 -> torchBFloat16.Full(shape, value, device)
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
        | Float16 -> torchFloat16.Random(shape, device)
        | BFloat16 -> torchBFloat16.Random(shape, device)
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
        | Float16 -> torchFloat16.RandomNormal(shape, device)
        | BFloat16 -> torchBFloat16.RandomNormal(shape, device)
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
        | Float16 -> torchFloat16.RandomInt(shape, low, high, device)
        | BFloat16 -> torchBFloat16.RandomInt(shape, low, high, device)
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
        | Float16 -> torchFloat16.CreateFromFlatArray(values, shape, device)
        | BFloat16 -> torchBFloat16.CreateFromFlatArray(values, shape, device)
        | Float32 -> torchFloat32.CreateFromFlatArray(values, shape, device)
        | Float64 -> torchFloat64.CreateFromFlatArray(values, shape, device)
        | Int8 -> torchInt8.CreateFromFlatArray(values, shape, device)
        | Byte -> torchByte.CreateFromFlatArray(values, shape, device)
        | Int16 -> torchInt16.CreateFromFlatArray(values, shape, device)
        | Int32 -> torchInt32.CreateFromFlatArray(values, shape, device)
        | Int64 -> torchInt64.CreateFromFlatArray(values, shape, device)
        | Bool -> torchBool.CreateFromFlatArray(values, shape, device)

