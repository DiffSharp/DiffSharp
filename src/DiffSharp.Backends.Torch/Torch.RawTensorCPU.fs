namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open TorchSharp
open TorchSharp.Tensor
open TorchSharp


#nowarn "77" // use of op_Explicit

type TorchShape = int64[]

[<AutoOpen>]
module internal Utils = 

    let torchScalarShape = [| |]
    let int64s (b: int[]) = Array.map int64 b

    let toTorchType dtype =
        match dtype with 
        | DType.Bool -> ATenScalarMapping.Byte
        | DType.Int8 -> ATenScalarMapping.SByte
        | DType.Int16 -> ATenScalarMapping.Short
        | DType.Int32 -> ATenScalarMapping.Int
        | DType.Int64 -> ATenScalarMapping.Long
        | DType.Float32 -> ATenScalarMapping.Float
        | DType.Float64 -> ATenScalarMapping.Double
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    let toTorchShape (shape: int[]) : TorchShape = int64s shape

    let toTorchDevice (device: Device) =
        match device with 
        | Device.CPU -> "cpu"
        | Device.GPU -> "gpu"
        | _ -> failwith "unknown device for Torch"

    let byteOfBool b = if b then 1uy else 0uy
    let boolOfByte b = (b <> 0uy)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    type RawTensor with
        member x.TorchTensor = (x :?> RawTensorTorch).TorchTensor

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type RawTensorTorch(tt: TorchTensor, shape: int[], dtype, device) =
    inherit RawTensor(shape, dtype, device, Backend.Torch)

    do 
       if tt.Type <> toTorchType dtype then
           failwithf "mismatched Torch tensor type, expected %A, got %A" (toTorchType dtype) tt.Type

       if toTorchShape shape <> tt.Shape then 
           failwithf "mismatched Torch tensor shape, expected %A, got %A" (toTorchShape shape) tt.Shape

    let clampBoolResult (result: TorchTensor) =
        match dtype with 
        | DType.Bool -> result.Clamp((0uy).ToScalar(), (1uy).ToScalar())
        | _ -> result

    member t.MakeLike(tt, ?shape, ?dtype) : RawTensor =
        upcast RawTensorTorch(tt, defaultArg shape t.Shape, defaultArg dtype t.DType, device)

    member x.TorchTensor = tt

    override t.GetSlice(fullBounds:int[,]) =
        let newShape = Shape.computeGetSlice fullBounds
        let mutable res = tt
        let mutable dim = 0 
        for i=0 to (fullBounds.GetLength(0) - 1) do
            let start = fullBounds.[i,0]
            let stop = fullBounds.[i,1] + 1

            let len = stop - start
            let idxs = LongTensor.Arange(int64 start, int64 stop, 1L)
            res <- res.IndexSelect(int64 dim, idxs)  // yield len // if len=1 then squeeze this dimension
            if fullBounds.[i, 2] = 1 && len = 1 then 
                res <- res.Squeeze(int64 dim)  // yield len // if len=1 then squeeze this dimension
            else
                dim <- dim + 1
        t.MakeLike(tt=res, shape=newShape)

    override t.Clone() =
        t.MakeLike(tt.Clone())

   // TODO: check if torch has a C++ hashing routine
    override x.ComputeHash() = 
        let mutable res = hash shape
        let n = shapeLength shape
        match dtype with 
        | DType.Int8 ->
            let data = tt.Data<sbyte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Bool ->
            let data = tt.Data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Int16 ->
            let data = tt.Data<int16>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i] )
        | DType.Int32 ->
            let data = tt.Data<int32>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Int64 -> 
            let data = tt.Data<int64>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Float32 ->
            let data = tt.Data<single>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        | DType.Float64 ->
            let data = tt.Data<double>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        | DType.Other _ -> failwith "Other types not supported by torch"
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
            | _ -> failwith "dim > 4"
        let obj = 
            match dtype with 
            | DType.Bool -> box (boolOfByte (item.DataItem<byte>()))
            | DType.Int8 -> box (item.DataItem<int8>())
            | DType.Int16 -> box (item.DataItem<int16>())
            | DType.Int32 -> box (item.DataItem<int32>())
            | DType.Int64 -> box (item.DataItem<int64>())
            | DType.Float32 -> box (item.DataItem<float32>())
            | DType.Float64 -> box (item.DataItem<double>())
            | _ -> failwith "Torch GetItem TBD type"
        obj

    member t.ToValuesTyped<'T, 'T2>(conv) : obj =
        match t.Shape with
        | [|  |] -> t.GetItem()
        | [| d0 |] -> upcast Array.init<'T> d0 (fun i -> tt.[int64 i].DataItem<'T2>() |> conv)
        | [| d0; d1 |] -> upcast Array2D.init<'T> d0 d1 (fun i j -> tt.[int64 i, int64 j].DataItem<'T2>() |> conv)
        | [| d0; d1; d2 |]  -> upcast Array3D.init<'T> d0 d1 d2 (fun i j k -> tt.[int64 i, int64 j, int64 k].DataItem<'T2>() |> conv)
        | [| d0; d1; d2; d3 |]  -> upcast Array4D.init<'T> d0 d1 d2 d3 (fun i j k l -> tt.[int64 i, int64 j, int64 k, int64 l].DataItem<'T2>() |> conv)
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override t.ToValues() =
        match dtype with 
        | DType.Bool -> t.ToValuesTyped<bool, byte>(boolOfByte)
        | DType.Int8 -> t.ToValuesTyped<sbyte, sbyte>(sbyte)
        | DType.Int16 -> t.ToValuesTyped<int16, int16>(id)
        | DType.Int32 -> t.ToValuesTyped<int32, int32>(id)
        | DType.Int64 -> t.ToValuesTyped<int64, int64>(id)
        | DType.Float32 -> t.ToValuesTyped<float32, float32>(id)
        | DType.Float64 -> t.ToValuesTyped<double, double>(id)
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    member t.ToRawData<'T>() : 'T[] =
        let data = tt.Data<'T>()
        let res = Array.zeroCreate<'T> (int32 tt.NumberOfElements)
        for i in 0 .. int32 tt.NumberOfElements - 1 do
            res.[i] <- data.[i]
        res

    member t.ToRawData() =
        match dtype with 
        | DType.Bool -> t.ToRawData<byte>() |> box
        | DType.Int8 -> t.ToRawData<sbyte>() |> box
        | DType.Int16 -> t.ToRawData<int16>() |> box
        | DType.Int32 -> t.ToRawData<int32>() |> box
        | DType.Int64 -> t.ToRawData<int64>() |> box
        | DType.Float32 -> t.ToRawData<float32>() |> box
        | DType.Float64 -> t.ToRawData<double>() |> box
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    override _.StackTs(tensors, dim) =
        let tts, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorTorch).TorchTensor, t.Shape) |> Array.unzip
        checkCanStack shapes dim
        let _n, _shape1, _shape2, newShape = Shape.computeStack shapes dim
        let result = tts.Stack(int64 dim)
        (tensors.[0] :?> RawTensorTorch).MakeLike(result, newShape)

    override t.UnstackT(dim) = 
        let shape = t.Shape
        let _shape1, _shape2, unstackedShape = Shape.computeUnstack shape dim
        let results = tt.Unbind(dim)
        results |> Array.map (fun rvalues -> t.MakeLike(rvalues, shape=unstackedShape))

    override t.CatTs(tensors, dim) = 
        let values, shapes = tensors |> Array.map (fun t -> t.TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _m2, _shape3, outShape = Shape.computeCat shapes dim
        let result = values.Cat(int64 dim)
        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.computeSplit shape sizes dim
        let results = tt.SplitWithSizes(int64s sizes, dim)
        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, shape=outShape))

    override t.TransposeT2() =
        checkCanTranspose t.Dim
        let newShape = Shape.computeTranspose t.Shape
        let result = tt.T()
        t.MakeLike(result, shape=newShape)

    override t.SqueezeT(dim) = 
        let shape = t.Shape
        let newShape = shapeSqueeze dim shape
        let mutable res = tt
        let mutable c = 0
        for i in 0 .. t.Dim - 1 do
            if shape.[i] = 1 && (dim = -1 || i = dim) then 
                res <- res.Squeeze(int64 c)
            else   
                c <- c + 1
        t.MakeLike(res, shape=newShape)

    override t.UnsqueezeT(dim) = 
        t.MakeLike(tt.Unsqueeze(int64 dim), shape=shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) = 
        let result = tt.Flip(int64s dims)
        t.MakeLike(result)

    override t.DilateT(dilations:int[]) = 
        checkCanDilate t.Dim dilations
        let outputShape = dilatedShape t.Shape dilations
        let shape4d = Array.append (Array.replicate (4 - t.Dim) 1) shape
        let dilations4d = Array.append (Array.replicate (4 - dilations.Length) 1) dilations
        let t4d = t.Expand(shape4d)
        let one = t.OneLike().TorchTensor
        let w2 = 
            let mutable w = t.ZerosLike(dilations4d).TorchTensor
            w.[0L,0L,0L,0L] <- one
            w

        match t.Dim with 
        | 1 ->
            let len0 = int64 shape.[0]
            let dilation0 = int64 dilations.[0]
            let res1 = t4d.TorchTensor.ConvTranspose1D(w2, stride=Nullable(dilation0))
            let lenOut = (len0 - 1L)*dilation0+1L 
            let res2 = res1.Slice(3L,0L,lenOut,1L)
            let res3 = res2.Reshape([| lenOut |])
            t.MakeLike(res3, outputShape)
        | 2 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let res1 = t4d.TorchTensor.ConvTranspose2D(w2, strides= [| dilation0; dilation1 |])
            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let res2 = res1.Slice(2L,0L,lenOut0,1L)
            let res3 = res2.Slice(3L,0L,lenOut1,1L)
            let res4 = res3.Reshape([| lenOut0; lenOut1 |])
            t.MakeLike(res4, outputShape)
        | 3 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let len2 = int64 shape.[2]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let dilation2 = int64 dilations.[2]
            if dilation0 <> 1L then 
                failwith "DilateT 3D not functioning correctly in LibTorch - RuntimeError: expected stride to be a single integer value or a list of 2 values to match the convolution dimensions, but got stride=[2, 2, 2]"
            // let res1 =  t4d.TorchTensor.ConvTranspose3D(w2, strides= [| dilation0; dilation1; dilation2 |])
            let results = 
                [| for i in 0 .. shape.[0]-1 do
                      let slice = t4d.TorchTensor.Slice(1L,int64 i, int64 (i+1), 1L)
                      yield slice.ConvTranspose2D(w2, strides= [| dilation1; dilation2 |]) |]
            let res1 = results.Cat(1L)

            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let lenOut2 = (len2 - 1L)*dilation2+1L 
            let res2 = res1.Slice(1L,0L,lenOut0,1L)
            let res3 = res2.Slice(2L,0L,lenOut1,1L)
            let res4 = res3.Slice(3L,0L,lenOut2,1L)
            let res5 = res4.Reshape([| lenOut0; lenOut1; lenOut2 |])
            t.MakeLike(res5, outputShape)
        | 4 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let len2 = int64 shape.[2]
            let len3 = int64 shape.[3]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let dilation2 = int64 dilations.[2]
            let dilation3 = int64 dilations.[3]
            if dilation0 <> 1L || dilation1 <> 1L then 
                failwith "DilateT 4D not easy in LibTorch unles dilation0 and dilation1 both 1"
            let res1 = 
                [| for i in 0 .. shape.[0]-1 do
                      [| for j in 0 .. shape.[1]-1 do
                            let slice = t4d.TorchTensor.Slice(0L,int64 i, int64 (i+1), 1L).Slice(1L,int64 j, int64 (j+1), 1L)
                            slice.ConvTranspose2D(w2, strides= [| dilation2; dilation3 |]) |].Cat(1L) |].Cat(0L)

            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let lenOut2 = (len2 - 1L)*dilation2+1L 
            let lenOut3 = (len3 - 1L)*dilation3+1L 
            let res2 = res1.Slice(0L,0L,lenOut0,1L)
            let res3 = res2.Slice(1L,0L,lenOut1,1L)
            let res4 = res3.Slice(2L,0L,lenOut2,1L)
            let res5 = res4.Slice(3L,0L,lenOut3,1L)
            let res5 = res5.Reshape([| lenOut0; lenOut1; lenOut2; lenOut3 |])
            t.MakeLike(res5, outputShape)
        | _ ->
            failwith "DilateT > 3D not available in LibTorch"

(*
import torch
import torch.nn.functional as F
def pad_within1d(x, stride):
  x2 = x.expand(1,1,1,x.size(0))
  w = x2.new_zeros(stride)
  w[0] = 1
  w2 = w.expand(1, 1, 1, stride)
  r = F.conv_transpose1d(x2, w2, stride=stride)
  return r[:,:,:,:-(stride-1)].reshape([(x.size(0)-1)*stride+1])

def pad_within2d(x, strides):
  x2 = x.expand(1,1,x.size(0),x.size(1))
  w = x2.new_zeros(strides[0], strides[1])
  w[0, 0] = 1
  w2 = w.expand(1, 1, strides[0], strides[1])
  r = F.conv_transpose2d(x2, w2, stride=strides)
  return r[:,:,:-(strides[0]-1),:-(strides[1]-1)].reshape([(x.size(0)-1)*strides[0]+1, (x.size(1)-1)*strides[1]+1])

def pad_within3d(x, strides):
  x2 = x.expand(1,x.size(0),x.size(1),x.size(2))
  w = x2.new_zeros(strides[0], strides[1], strides[2])
  w[0, 0, 0] = 1
  w2 = w.expand(1, strides[0], strides[1], strides[2])
  r = F.conv_transpose3d(x2, w2, stride=strides)
  return r[:,:-(strides[0]-1),:-(strides[1]-1),:-(strides[2]-1)].reshape([(x.size(0)-1)*strides[0]+1, (x.size(1)-1)*strides[1]+1, (x.size(2)-1)*strides[2]+1])

pad_within1d(torch.tensor([6,1,2,3]),2)
pad_within1d(torch.tensor([6,1,2,3]),3)
pad_within2d(torch.tensor([[6,1,2,3],[7,1,2,3],[8,1,2,3]]),[2,3])
pad_within3d(torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]]),[2,2,2])        
*)

    override t.UndilateT(dilations:int[]) =
        let outputShape = undilatedShape t.Shape dilations
        let mutable res = tt
        for d in 0 .. dilations.Length - 1 do
            res <- res.Slice(int64 d, 0L, int64 shape.[d], int64 dilations.[d])
        t.MakeLike(res, outputShape)

    override t.ViewT(shape:int[]) =
        checkCanView t.Shape shape
        t.MakeLike(tt.View(toTorchShape shape), shape=shape)

    override t.Cast(newDType: DType) =
        if newDType = t.DType then 
            upcast t
        else 
            let result = tt.ToType(toTorchType newDType)
            t.MakeLike(result, dtype=newDType)

    override _.RandomMultinomial(numSamples) =
        failwith "tbd"

    override _.Equals(t2:RawTensor) : bool = 
        if dtype = t2.DType then
            let r1 = (shape = t2.Shape)
            if not r1 then false else
            let tt2 = t2.TorchTensor
            let r2 = tt.Equal(tt2)
            r2
        else 
            opNotSupported2 "Equals" dtype t2.DType

    override t.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        if dtype = t2.DType then
            match dtype with 
            | DType.IntegralOrBool -> t.Equals(t2)
            | _ -> tt.AllClose(t2.TorchTensor, relativeTolerance, absoluteTolerance)
        else 
            opNotSupported2 "Equals" dtype t2.DType
        

    override t1.LtTT(t2) =
        let result = tt.Lt(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.GtTT(t2) =
        let result = tt.Gt(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.LeTT(t2) = 
        let result = tt.Le(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.GeTT(t2) = 
        let result = tt.Ge(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.EqTT(t2) = 
        let result = tt.Eq(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.NeqTT(t2) = 
        let result = tt.Ne(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

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
        | DType.Bool -> t.Cast(DType.Int8).MinIndexT() // TODO: could likely be improved
        | _ -> t.NegT().MaxIndexT()

    override t1.AddTT(t2) =
        let result = tt.Add(t2.TorchTensor) |> clampBoolResult
        t1.MakeLike(result)

    override t1.AddTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Add(t2v) |> clampBoolResult
        t1.MakeLike(result)

    override t1.AddT2T1(t2) = 
        let result = tt.Add(t2.TorchTensor) |> clampBoolResult
        t1.MakeLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = shapeUnsqueezeAs shape2 shape1
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
        | DType.Bool -> opNotSupported2 "SubT" t1.DType t2.DType
        | _ ->
        let result = tt.Sub(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.SubT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v - t2.TorchTensor
        (t2 :?> RawTensorTorch).MakeLike(result)

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
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let result = tt.Div(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.DivT0T(t2) =
        match dtype with 
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let t1v = t1.TorchTensor.Item()
        let result = t1v / t2.TorchTensor
        (t2 :?> RawTensorTorch).MakeLike(result)

    override t1.DivTT0(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let t2v = t2.TorchTensor.Item()
        let result = tt.Div(t2v)
        t1.MakeLike(result)

    override t1.PowTT(t2) =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowTT" dtype
        | _ -> 
        let result = tt.Pow(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.PowT0T(t2) = 
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowT0T" dtype
        | _ -> 
        let result = t1.Expand(t2.Shape).TorchTensor.Pow(t2.TorchTensor)
        (t2 :?> RawTensorTorch).MakeLike(result)

    override t1.PowTT0(t2) =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowTT0" dtype
        | _ -> 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Pow(t2v)
        t1.MakeLike(result)

    override t1.MatMulT2T2(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "MatMulT2T2" t1.DType t2.DType
        | _ ->  
        checkCanMatmul t1.Shape t2.Shape
        let result = tt.Mm(t2.TorchTensor)
        t1.MakeLike(result, [| t1.Shape.[0]; t2.Shape.[1] |])

    /// Wraps a convolution operation - integer types get promoted to floats. LibTorch only appears to implement
    /// floating point convolutions (as of Torch 1.0.1)
    static member WrapConv (t1: RawTensorTorch) (t2: RawTensor) f =
        let tt1, tt2 = t1.TorchTensor, t2.TorchTensor
        let tt1, tt2 =
            match t1.DType with 
            | DType.Bool -> opNotSupported2 "Conv1D" t1.DType t2.DType
            | DType.Int8 | DType.Int16 -> tt1.ToType(ATenScalarMapping.Float), tt2.ToType(ATenScalarMapping.Float)
            | DType.Int32 | DType.Int64 -> tt1.ToType(ATenScalarMapping.Double), tt2.ToType(ATenScalarMapping.Double)
            | DType.Float32 | DType.Float64 -> tt1, tt2
            | DType.Other _ -> failwith "Conv1D - other type"
        let result : TorchTensor = f (tt1, tt2)
        let result2 =
            match t1.DType with 
            | DType.Bool -> opNotSupported2 "Conv1D" t1.DType t2.DType
            | DType.Int32 | DType.Int64
            | DType.Int8 | DType.Int16 -> result.ToType(toTorchType t1.DType)
            | DType.Float32 | DType.Float64 -> result
            | DType.Other _ -> failwith "Conv1D - other type"
        result2

    override t1.Conv1D(t2, stride, padding) = // TODO: bias, dilation and groups
        let _outputLength, outputShape = Shape.computeConv1D t1.Shape t2.Shape stride padding
        let result =
            RawTensorTorch.WrapConv t1 t2 (fun (tt1, tt2) ->
                tt1.Conv1D(tt2, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L)))
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t1.Conv2D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _outputHeight, _outputWidth, outputShape = Shape.computeConv2D t1.Shape t2.Shape strides paddings
        let result = 
            RawTensorTorch.WrapConv t1 t2 (fun (tt1, tt2) ->
                tt1.Conv2D(tt2, strides=int64s strides, padding=int64s paddings))
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t1.Conv3D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _outputDepth, _outputHeight, _outputWidth, outputShape = Shape.computeConv3D t1.Shape t2.Shape strides paddings
        let result = 
            RawTensorTorch.WrapConv t1 t2 (fun (tt1, tt2) ->
                tt1.Conv3D(tt2, strides=int64s strides, padding=int64s paddings))
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t.SumT2Dim0() =
        let result = tt.Sum([| 0L |], ``type``= Nullable(tt.Type))
        let resultShape = [|t.Shape.[1]|]
        t.MakeLike(result, shape=resultShape)

    override t.NegT() =
        match dtype with 
        | DType.Bool -> opNotSupported "NegT" t.DType
        | _ ->  t.MakeLike(-tt)

    override t.SumT(?resultType) =
        let typeArg = match resultType with None -> Nullable() | Some dt -> Nullable(toTorchType dt)
        let outType = match resultType with None -> dtype.SummationType | Some dt -> dt
        t.MakeLike(tt.Sum(typeArg), shape=Shape.scalar, dtype=outType)

    override t.SignT() =
        t.MakeLike(tt.Sign())

    override t.FloorT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "FloorT" t.DType
        | _ ->  t.MakeLike(tt.Floor())

    override t.CeilT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CeilT" t.DType
        | _ ->  t.MakeLike(tt.Ceil())

    override t.RoundT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "RoundT" t.DType
        | _ ->  t.MakeLike(tt.Round())

    override t.AbsT() = 
        match dtype with 
        | DType.Bool -> opNotSupported "AbsT" t.DType
        | DType.Int8 -> t.Cast(DType.Int32).AbsT().Cast(DType.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ -> t.MakeLike(tt.Abs ())

    override t.SoftplusT() = 
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SoftplusT" t.DType
        | _ -> t.MakeLike(tt.Softplus())

    override t.ReluT() =
        match dtype with 
        | DType.Bool -> opNotSupported "ReluT" t.DType
        | DType.Int8 -> t.Cast(DType.Int32).ReluT().Cast(DType.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ ->   t.MakeLike(tt.Relu())

    override t.SigmoidT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SigmoidT" t.DType
        | _ ->  t.MakeLike(tt.Sigmoid())

    override t.ExpT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "ExpT" t.DType
        | _ ->  t.MakeLike(tt.Exp())

    override t.LogT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "LogT" t.DType
        | _ ->  t.MakeLike(tt.Log())

    override t.Log10T() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "Log10T" t.DType
        | _ ->   t.MakeLike(tt.Log10())

    override t.SqrtT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SqrtT" t.DType
        | _ ->  t.MakeLike(tt.Sqrt())

    override t.SinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SinT" t.DType
        | _ ->  t.MakeLike(tt.Sin())

    override t.CosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CosT" t.DType
        | _ ->  t.MakeLike(tt.Cos())

    override t.TanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "TanT" t.DType
        | _ ->  t.MakeLike(tt.Tan())

    override t.SinhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SinhT" t.DType
        | _ ->  t.MakeLike(tt.Sinh())

    override t.CoshT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CoshT" t.DType
        | _ ->  t.MakeLike(tt.Cosh())

    override t.TanhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "TanhT" t.DType
        | _ ->  t.MakeLike(tt.Tanh())

    override t.AsinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AsinT" t.DType
        | _ ->  t.MakeLike(tt.Asin())

    override t.AcosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AcosT" t.DType
        | _ ->  t.MakeLike(tt.Acos())

    override t.AtanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AtanT" t.DType
        | _ ->  t.MakeLike(tt.Atan())

    new (info: System.Runtime.Serialization.SerializationInfo, _context: System.Runtime.Serialization.StreamingContext) =
        let device = info.GetValue("device", typeof<Device>) :?> Device
        let dtype = info.GetValue("dtype", typeof<DType>) :?> DType
        let shape = info.GetValue("shape", typeof<Shape>) :?> Shape
        let tt =
            match dtype with 
            | DType.Bool -> 
                let data = info.GetValue("data", typeof<byte[]>)  :?> byte[]
                ByteTensor.From (data, toTorchShape shape) 
            | DType.Int8 -> 
                let data = info.GetValue("data", typeof<byte[]>)  :?> sbyte[]
                SByteTensor.From (data, toTorchShape shape) 
            | DType.Int16 -> 
                let data = info.GetValue("data", typeof<int16[]>)  :?> int16[]
                ShortTensor.From (data, toTorchShape shape) 
            | DType.Int32 -> 
                let data = info.GetValue("data", typeof<int32[]>)  :?> int32[]
                IntTensor.From (data, toTorchShape shape) 
            | DType.Int64 -> 
                let data = info.GetValue("data", typeof<int64[]>)  :?> int64[]
                LongTensor.From (data, toTorchShape shape) 
            | DType.Float32 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                FloatTensor.From (data, toTorchShape shape) 
            | DType.Float64 -> 
                let data = info.GetValue("data", typeof<double[]>)  :?> double[]
                DoubleTensor.From (data, toTorchShape shape) 
            | DType.Other _ -> failwith "deserialize other type in torch nyi"

        RawTensorTorch(tt, shape, dtype, device)

    interface System.Runtime.Serialization.ISerializable with

        //[SecurityPermissionAttribute(SecurityAction.Demand,  SerializationFormatter = true)]
        member t.GetObjectData(info, _context) =
            info.AddValue("device", device)
            info.AddValue("dtype", dtype)
            info.AddValue("shape", shape)
            info.AddValue("data", t.ToRawData())

/// The concrete implementation of RawTensorStatics for Float32 data.
type TorchStatics<'T, 'T2>
       (dtype: DType, device: Device, conv: 'T -> 'T2,
        from0: 'T2 -> TorchTensor,
        from: 'T2[] * TorchShape -> TorchTensor,
        zero: 'T, one: 'T,
        zeros: TorchShape  * string -> TorchTensor,
        ones: TorchShape  * string -> TorchTensor,
        random: TorchShape  * string -> TorchTensor,
        randomN: TorchShape  * string -> TorchTensor,
        randomIntegers: int64 * TorchShape * string -> TorchTensor,
        valueFromObj: obj -> 'T,
        scalarFromConvValue: 'T2 -> Scalar) = 

    inherit RawTensorStatics()
    let torchDevice = toTorchDevice device

    override _.Zero =
        let tt = from0(conv(zero)).[0L]
        RawTensorTorch(tt, Shape.scalar, dtype, device) :> _ 
    override _.One =
        let tt = from0(conv(one)).[0L]
        RawTensorTorch(tt, Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:int[]) = RawTensorTorch(zeros(toTorchShape shape, torchDevice), shape, dtype, device) :> _
    override _.Ones(shape:int[]) = RawTensorTorch(ones(toTorchShape shape, torchDevice), shape, dtype, device) :> _
    override _.Random(shape:int[]) = RawTensorTorch(random(toTorchShape shape, torchDevice), shape, dtype, device) :> _
    override _.RandomNormal(shape:int[]) = RawTensorTorch(randomN(toTorchShape shape, torchDevice), shape, dtype, device) :> _
    override _.RandomIntegers(maxn, shape:int[]) = RawTensorTorch(randomIntegers(maxn, toTorchShape shape, torchDevice), shape, dtype, device) :> _

    override _.Full(shape:int[], value:obj) =
        let t = zeros(toTorchShape shape, torchDevice)
        t.FillInPlace(scalarFromConvValue (conv (valueFromObj value))) |> ignore
        RawTensorTorch(t, shape, dtype, device) :> _

    override ts.CreateFromFlatArray(values:Array, shape) =
        let values = values :?> 'T[] |> Array.map conv 
        let t = 
            match shape with 
            | [| |] -> from0(values.[0]).[0L]
            | _ -> from (values, toTorchShape shape)
        RawTensorTorch(t, shape, dtype, device) :> _

/// The concrete implementation of RawTensorStatics for Bool CPU data.
type RawTensorFloat32CPUStatics() = 

    inherit TorchStatics<single, single>(DType.Float32, Device.CPU, id, 
        (fun v -> FloatTensor.From(v)), 
        FloatTensor.From, 
        0.0f, 1.0f, 
        (fun (shape, device) -> FloatTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.Ones(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.Random(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.RandomN(shape, device=device)), 
        (fun (max, shape, device) -> FloatTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToSingle, 
        Scalar.op_Implicit)

type RawTensorFloat64CPUStatics() = 

    inherit TorchStatics<double, double>(DType.Float64, Device.CPU, id, 
        (fun v -> DoubleTensor.From(v)), 
        DoubleTensor.From, 
        0.0, 1.0, 
        (fun (shape, device) -> DoubleTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.Ones(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.Random(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.RandomN(shape, device=device)), 
        (fun (max, shape, device) -> DoubleTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToDouble, 
        Scalar.op_Implicit)

type RawTensorByteCPUStatics() = 

    inherit TorchStatics<byte, byte>(DType.Int8, Device.CPU, byte,
        (fun v -> ByteTensor.From(v)), 
        ByteTensor.From, 
        0uy, 1uy,
        (fun (shape, device) -> ByteTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> ByteTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int8), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int8), 
        (fun (max, shape, device) -> ByteTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToByte, 
        Scalar.op_Implicit)

type RawTensorInt8CPUStatics() = 

    inherit TorchStatics<sbyte, sbyte>(DType.Int8, Device.CPU, sbyte,
        (fun v -> SByteTensor.From(v)), 
        SByteTensor.From, 
        0y, 1y,
        (fun (shape, device) -> SByteTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> SByteTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int8), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int8), 
        (fun (max, shape, device) -> SByteTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToSByte, 
        Scalar.op_Implicit)

type RawTensorInt16CPUStatics() = 

    inherit TorchStatics<int16, int16>(DType.Int16, Device.CPU, int16, 
        (fun v -> ShortTensor.From(v)), 
        ShortTensor.From, 
        0s, 1s,
        (fun (shape, device) -> ShortTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> ShortTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int16), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int16), 
        (fun (max, shape, device) -> ShortTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToInt16, 
        Scalar.op_Implicit)

type RawTensorInt32CPUStatics() = 

    inherit TorchStatics<int32, int32>(DType.Int32, Device.CPU, int32, 
        (fun v -> IntTensor.From(v)), 
        IntTensor.From, 
        0, 1,
        (fun (shape, device) -> IntTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> IntTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int32), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int32), 
        (fun (max, shape, device) -> IntTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToInt32, 
        Scalar.op_Implicit)

type RawTensorInt64CPUStatics() = 

    inherit TorchStatics<int64, int64>(DType.Int64, Device.CPU, int64, 
        (fun v -> LongTensor.From(v)), 
        LongTensor.From, 
        0L, 1L,
        (fun (shape, device) -> LongTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> LongTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int64), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int64), 
        (fun (max, shape, device) -> LongTensor.RandomIntegers(max, shape, device=device)), 
        System.Convert.ToInt64, 
        Scalar.op_Implicit)

type RawTensorBoolCPUStatics() = 

    inherit TorchStatics<bool, byte>(DType.Bool, Device.CPU, byteOfBool, 
        (fun v -> ByteTensor.From(v)), 
        ByteTensor.From, 
        false, true,
        (fun (shape, device) -> ByteTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> ByteTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Bool), 
        (fun _ -> opNotSupported "RandomNormal"  DType.Bool), 
        (fun (maxn, shape, device) -> ByteTensor.RandomIntegers(min 2L maxn, shape, device=device)), 
        System.Convert.ToBoolean, 
        Scalar.op_Implicit)
