namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
//open AtenSharp
open TorchSharp
open TorchSharp.Tensor


#nowarn "77" // use of op_Explicit

type TorchShape = int64[]

[<AutoOpen>]
module internal Utils = 
    let opNotSupported (t: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type %A" t)

    let opNotSupported2 (t1: DType) (t2: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type (%A, %A)" t1 t2)

    let torchScalarShape = [| |]
    let int64s (b: int[]) = Array.map int64 b
    let toTorchShape (shape: int[]) = (* if shape.Length = 0 then torchScalarShape else *) int64s shape
    let byteOfBool b = if b then 1uy else 0uy
    let boolOfByte b = (b <> 0uy)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    type RawTensor with
        member x.TorchTensor = (x :?> RawTensorTorch).TorchTensor

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type RawTensorTorch(tt: TorchTensor, shape: int[], dtype, device) =
    inherit RawTensor(shape, dtype, device, Backend.Torch)

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
        | DType.Int8
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
            | DType.Bool -> box (item.DataItem<byte>() = 0uy)
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
        | DType.Int8 -> t.ToValuesTyped<sbyte, byte>(sbyte)
        | DType.Int16 -> t.ToValuesTyped<int16, int16>(id)
        | DType.Int32 -> t.ToValuesTyped<int32, int32>(id)
        | DType.Int64 -> t.ToValuesTyped<int64, int64>(id)
        | DType.Float32 -> t.ToValuesTyped<float32, float32>(id)
        | DType.Float64 -> t.ToValuesTyped<double, double>(id)
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
        let mutable w = t.ZerosLike(dilations).TorchTensor
        let one = t.OneLike().TorchTensor
        match dilations.Length with
        | 1 -> w.[0L] <- one
        | 2 -> w.[0L,0L] <- one
        | 3 -> w.[0L,0L,0L] <- one
        | _ -> failwith "DilateT 4D"
        let w2 = w.Expand(int64s dilations4d)
        match t.Dim with 
        | 1 ->
            let stride = int64 dilations.[0]
            let bias = Nullable(t.ZerosLike([|shape.[0]|]).TorchTensor)
            let r = t4d.TorchTensor.ConvTranspose1D(w2, bias=bias, stride=Nullable(stride))
            let res = r.Slice(3L,0L,stride-1L,1L).Reshape([| (int64 (shape.[0]-1))*stride+1L |])
            t.MakeLike(res, outputShape)
        | _ -> failwith "DilateT 4D"
           

        //let preShape = dilatedShape2 t.Shape dilations
        //let t3 = t.StackTs([| yield t; for i in 1..dilations.[0]-1 do yield t.ZerosLike(t.Shape) |], 1).ViewT(preShape)
        //let tt3 = t3.TorchTensor
        //let ttres = tt3.Narrow(0L, 0L, int64 outputShape.[0])
        //t.MakeLike(ttres, outputShape)

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
        failwith "TBD - UndilateT"
        //match t.Dim with
        //| 0 -> t.Clone()
        //| _ ->
        //    let result = t.ZerosLike(undilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
        //    let rec dilate (shape:int[]) externalCoords = 
        //        if shape.Length = 1 then
        //            for i=0 to shape.[0]-1 do
        //                let globalCoords = Array.append externalCoords [|i|]
        //                result.[globalCoords] <- t.[dilatedCoordinates globalCoords dilations]
        //        else
        //            for i=0 to shape.[0]-1 do
        //                dilate shape.[1..] (Array.append externalCoords [|i|])
        //    dilate result.Shape [||]        
        //    upcast result        

    override t.ViewT(shape:int[]) =
        checkCanView t.Shape shape
        t.MakeLike(tt.View(toTorchShape shape), shape=shape)

    override t.Cast(dtype: DType) =
        if dtype = t.DType then 
            upcast t
        else 
            let atenType = 
                match dtype with 
                | DType.Bool -> ATenScalarMapping.Byte
                | DType.Int8 -> ATenScalarMapping.Byte
                | DType.Int16 -> ATenScalarMapping.Short
                | DType.Int32 -> ATenScalarMapping.Int
                | DType.Int64 -> ATenScalarMapping.Long
                | DType.Float32 -> ATenScalarMapping.Float
                | DType.Float64 -> ATenScalarMapping.Double
                | DType.Other _ -> failwith "Torch GetItem TBD other type"
            let result = tt.ToType(atenType)
            t.MakeLike(result, dtype=dtype)

    override _.RandomMultinomial(numSamples) =
        failwith "tbd"

    override _.Equals(t2:RawTensor) : bool = 
        (shape = t2.Shape) && tt.Equal(t2.TorchTensor)

    override _.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        tt.AllClose(t2.TorchTensor, relativeTolerance, absoluteTolerance)

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
    override t.MinIndexT() = t.NegT().MaxIndexT()

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
        let result = tt.Div(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.DivT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v / t2.TorchTensor
        (t2 :?> RawTensorTorch).MakeLike(result)

    override t1.DivTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Div(t2v)
        t1.MakeLike(result)

    override t1.PowTT(t2) =
        let result = tt.Pow(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.PowT0T(t2) = 
        let result = t1.Expand(t2.Shape).TorchTensor.Pow(t2.TorchTensor)
        (t2 :?> RawTensorTorch).MakeLike(result)

    override t1.PowTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Pow(t2v)
        t1.MakeLike(result)

    override t1.MatMulT2T2(t2) = 
        checkCanMatmul t1.Shape t2.Shape
        let result = tt.Mm(t2.TorchTensor)
        t1.MakeLike(result, [| t1.Shape.[0]; t2.Shape.[1] |])

    override t1.Conv1D(t2, stride, padding) = // TODO: bias, dilation and groups
        let outputLength, outputShape = Shape.computeConv1D t1.Shape t2.Shape stride padding
        let bias = Nullable(t1.ZerosLike([|outputLength|]).TorchTensor)
        let result = tt.Conv1D(t2.TorchTensor, bias=bias, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L))
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t1.Conv2D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let outputHeight, outputWidth, outputShape = Shape.computeConv2D t1.Shape t2.Shape strides paddings
        let bias = t1.ZerosLike([|outputHeight; outputWidth|]).TorchTensor
        let result = tt.Conv2D(t2.TorchTensor, bias=Nullable(bias), strides=int64s strides, paddings=int64s paddings)
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t1.Conv3D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let outputDepth, outputHeight, outputWidth, outputShape = Shape.computeConv3D t1.Shape t2.Shape strides paddings
        let bias = t1.ZerosLike([|outputDepth; outputHeight; outputWidth|]).TorchTensor
        let result = tt.Conv3D(t2.TorchTensor, bias=Nullable(bias), strides=int64s strides, paddings=int64s paddings)
        (t2 :?> RawTensorTorch).MakeLike(result, shape=outputShape)

    override t.SumT2Dim0() =
        let result = tt.Sum([| 0L |])
        let resultShape = [|t.Shape.[1]|]
        t.MakeLike(result, shape=resultShape)

    override t.NegT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.MakeLike(-tt)

    override t.SumT() =
        match dtype with 
        | DType.Bool -> t.Cast(Int64).SumT()
        | _ ->  t.MakeLike(tt.Sum(), shape=Shape.scalar)

    override t.SignT() =
        //match dtype with 
        //| DType.Bool -> opNotSupported t.DType
        //| _ ->  
        t.MakeLike(tt.Sign())

    override t.FloorT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Floor())

    override t.CeilT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Ceil())

    override t.RoundT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Round())

    override t.AbsT() = 
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ -> t.MakeLike(tt.Abs ())

    override t.SoftplusT() = 
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ -> t.MakeLike(tt.Softplus())

    override t.ReluT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->   t.MakeLike(tt.Relu())

    override t.SigmoidT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Sigmoid())

    override t.ExpT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Exp())

    override t.LogT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Log())

    override t.Log10T() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->   t.MakeLike(tt.Log10())

    override t.SqrtT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Sqrt())

    override t.SinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Sin())

    override t.CosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Cos())

    override t.TanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Tan())

    override t.SinhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Sinh())

    override t.CoshT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Cosh())

    override t.TanhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Tanh())

    override t.AsinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Asin())

    override t.AcosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Acos())

    override t.AtanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.MakeLike(tt.Atan())

/// The concrete implementation of RawTensorStatics for Float32 data.
type TorchStatics<'T, 'T2>
       (dtype: DType, device: Device, conv: 'T -> 'T2,
        from0: 'T2 -> TorchTensor,
        from: 'T2[] * TorchShape -> TorchTensor,
        zero: 'T, one: 'T,
        zeros: TorchShape -> TorchTensor,
        ones: TorchShape -> TorchTensor,
        random: TorchShape -> TorchTensor,
        randomN: TorchShape -> TorchTensor,
        valueFromObj: obj -> 'T,
        scalarFromConvValue: 'T2 -> Scalar) = 

    inherit RawTensorStatics()

    override _.Zero = RawTensorTorch(from0(conv(zero)).[0L], Shape.scalar, dtype, device) :> _ 
    override _.One =  RawTensorTorch(from0(conv(one)).[0L], Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:int[]) = RawTensorTorch(zeros(toTorchShape shape), shape, dtype, device) :> _
    override _.Ones(shape:int[]) = RawTensorTorch(ones(toTorchShape shape), shape, dtype, device) :> _
    override _.Random(shape:int[]) = RawTensorTorch(random(toTorchShape shape), shape, dtype, device) :> _
    override _.RandomNormal(shape:int[]) = RawTensorTorch(randomN(toTorchShape shape), shape, dtype, device) :> _

    override _.Full(shape:int[], value:obj) =
        let t = zeros(toTorchShape shape)
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

    inherit TorchStatics<single, single>(DType.Float32, Device.CPU, id, FloatTensor.From, FloatTensor.From, 0.0f, 1.0f, FloatTensor.Zeros, FloatTensor.Ones, FloatTensor.Random, FloatTensor.RandomN, System.Convert.ToSingle, Scalar.op_Implicit)

type RawTensorFloat64CPUStatics() = 

    inherit TorchStatics<double, double>(DType.Float64, Device.CPU, id, DoubleTensor.From, DoubleTensor.From, 0.0, 1.0, DoubleTensor.Zeros, DoubleTensor.Ones, DoubleTensor.Random, DoubleTensor.RandomN, System.Convert.ToDouble, Scalar.op_Implicit)

type RawTensorInt8CPUStatics() = 

    inherit TorchStatics<int8, byte>(DType.Int8, Device.CPU, byte, ByteTensor.From, ByteTensor.From, 0y, 1y, ByteTensor.Zeros, ByteTensor.Ones, ByteTensor.Random, ByteTensor.RandomN, System.Convert.ToSByte, Scalar.op_Implicit)

type RawTensorInt16CPUStatics() = 

    inherit TorchStatics<int16, int16>(DType.Int16, Device.CPU, int16, ShortTensor.From, ShortTensor.From, 0s, 1s, ShortTensor.Zeros, ShortTensor.Ones, ShortTensor.Random, ShortTensor.RandomN, System.Convert.ToInt16, Scalar.op_Implicit)

type RawTensorInt32CPUStatics() = 

    inherit TorchStatics<int32, int32>(DType.Int32, Device.CPU, int32, IntTensor.From, IntTensor.From, 0, 1, IntTensor.Zeros, IntTensor.Ones, IntTensor.Random, IntTensor.RandomN, System.Convert.ToInt32, Scalar.op_Implicit)

type RawTensorInt64CPUStatics() = 

    inherit TorchStatics<int64, int64>(DType.Int64, Device.CPU, int64, LongTensor.From, LongTensor.From, 0L, 1L, LongTensor.Zeros, LongTensor.Ones, LongTensor.Random, LongTensor.RandomN, System.Convert.ToInt64, Scalar.op_Implicit)

type RawTensorBoolCPUStatics() = 

    inherit TorchStatics<bool, byte>(DType.Bool, Device.CPU, byteOfBool, ByteTensor.From, ByteTensor.From, false, true, ByteTensor.Zeros, ByteTensor.Ones, ByteTensor.Random, ByteTensor.RandomN, System.Convert.ToBoolean, Scalar.op_Implicit)
