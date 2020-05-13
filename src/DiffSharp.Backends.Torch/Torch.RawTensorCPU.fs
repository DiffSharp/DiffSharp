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
    let toTorchShape (shape: int[]) = (* if shape.Length = 0 then torchScalarShape else *) Array.map int64 shape
    let byteOfBool b = if b then 1uy else 0uy
    let boolOfByte b = (b <> 0uy)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    type RawTensor with
        member x.TorchTensor = (x :?> TorchTensorCPU).TorchTensor

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type TorchTensorCPU(tt: TorchTensor, shape: int[], dtype, device) =
    inherit RawTensor(shape, dtype, device, Backend.Torch)

    member t.CreateLike(tt, ?shape, ?dtype) : RawTensor =
        upcast TorchTensorCPU(tt, defaultArg shape t.Shape, defaultArg dtype t.DType, device)

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
        t.CreateLike(tt=res, shape=newShape)

    override t.Clone() =
        t.CreateLike(tt.Clone())

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
        t.CreateLike(tt.Expand(toTorchShape newShape), shape=newShape)

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
        let _tts, shapes = tensors |> Array.map (fun t -> (t :?> TorchTensorCPU).TorchTensor, t.Shape) |> Array.unzip
        checkCanStack shapes dim
        let _n, _shape1, _shape2, newShape = Shape.computeStack shapes dim
        let result = failwith "tbd" //tts.[0].
        (tensors.[0] :?> TorchTensorCPU).CreateLike(result, newShape)

    override t.UnstackT(dim) = 
        let shape = t.Shape
        let _shape1, _shape2, unstackedShape = Shape.computeUnstack shape dim
        let results = tt.Unbind(dim)
        results |> Array.map (fun rvalues -> t.CreateLike(rvalues, shape=unstackedShape))

    override t.CatTs(tensors, dim) = 
        let values, shapes = tensors |> Array.map (fun t -> t.TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _m2, _shape3, outShape = Shape.computeCat shapes dim
        let result = values.Cat(int64 dim)
        t.CreateLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.computeSplit shape sizes dim
        let results = tt.SplitWithSizes(Array.map int64 sizes)
        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.CreateLike(rvalues, shape=outShape))

    override t.TransposeT2() = failwith "TBD - TransposeT2"
        //checkCanBatchTranspose t.Dim
        //let oldShape = t.Shape
        //let batch = oldShape.[0..oldShape.Length-3]
        //let nrows = oldShape.[oldShape.Length-2]
        //let ncols = oldShape.[oldShape.Length-1]
        //let newShape = Array.append batch [| ncols; nrows |]
        //let result = Array.zeroCreate values.Length
        //for i = 0 to values.Length-1 do
        //    let col = i % ncols 
        //    let row = (i / ncols ) % nrows
        //    let j = (i / ncols / nrows)*ncols*nrows + col*nrows + row
        //    result.[j] <- values.[i]
        //t.CreateLike(result, newShape)

    override t.SqueezeT(dim) = 
        t.CreateLike(tt.Squeeze(int64 dim), shape=shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) = 
        t.CreateLike(tt.Unsqueeze(int64 dim), shape=shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) = failwith "TBD - FlipT"
        //checkCanFlip t.Dim dims
        //match t.Dim with
        //| 0 -> t.Clone()
        //| _ ->
        //    let result = t.ZerosLike(t.Shape) :?> RawTensorCPU<'T>
        //    let rec flip (shape:int[]) externalCoords = 
        //        if shape.Length = 1 then
        //            for i=0 to shape.[0]-1 do
        //                let globalCoords = Array.append externalCoords [|i|]
        //                result.[mirrorCoordinates globalCoords t.Shape dims] <- t.[globalCoords]
        //        else
        //            for i=0 to shape.[0]-1 do
        //                flip shape.[1..] (Array.append externalCoords [|i|])
        //    flip t.Shape [||]        
        //    upcast result

    override t.DilateT(dilations:int[]) = failwith "TBD - DilateT"
        //checkCanDilate t.Dim dilations
        //match t.Dim with
        //| 0 -> t.Clone()
        //| _ ->
        //    let result = t.ZerosLike(dilatedShape t.Shape dilations) :?> RawTensorCPU<'T>
        //    let rec dilate (shape:int[]) externalCoords = 
        //        if shape.Length = 1 then
        //            for i=0 to shape.[0]-1 do
        //                let globalCoords = Array.append externalCoords [|i|]
        //                result.[dilatedCoordinates globalCoords dilations] <- t.[globalCoords]
        //        else
        //            for i=0 to shape.[0]-1 do
        //                dilate shape.[1..] (Array.append externalCoords [|i|])
        //    dilate t.Shape [||]        
        //    upcast result        

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
        t.CreateLike(tt.View(toTorchShape shape), shape=shape)

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
            t.CreateLike(result, dtype=dtype)

    override _.RandomMultinomial(numSamples) =
        failwith "tbd"

    override _.Equals(t2:RawTensor) : bool = 
        (shape = t2.Shape) && tt.Equal(t2.TorchTensor)

    override _.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        tt.AllClose(t2.TorchTensor, relativeTolerance, absoluteTolerance)

    override t.SoftplusT() = 
        t.CreateLike(tt.Softplus(), dtype=DType.Bool)

    override t1.LtTT(t2) =
        let result = tt.Lt(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

    override t1.GtTT(t2) =
        let result = tt.Gt(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

    override t1.LeTT(t2) = 
        let result = tt.Le(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

    override t1.GeTT(t2) = 
        let result = tt.Ge(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

    override t1.EqTT(t2) = 
        let result = tt.Eq(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

    override t1.NeqTT(t2) = 
        let result = tt.Ne(t2.TorchTensor)
        t1.CreateLike(result, dtype=DType.Bool)

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
        t1.CreateLike(result)

    override t1.AddTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Add(t2v)
        t1.CreateLike(result)

    override t1.AddT2T1(t2) = 
        let result = tt.Add(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        failwith "tbd AddTTSlice" 

    override t1.SubTT(t2) = 
        let result = tt.Sub(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.SubT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v - t2.TorchTensor
        t1.CreateLike(result)

    override t1.SubTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Sub(t2v)
        t1.CreateLike(result)

    override t1.MulTT(t2) = 
        let result = tt.Mul(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.MulTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Mul(t2v)
        t1.CreateLike(result)

    override t1.DivTT(t2) = 
        let result = tt.Div(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.DivT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v / t2.TorchTensor
        t1.CreateLike(result)

    override t1.DivTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Div(t2v)
        t1.CreateLike(result)

    override t1.PowTT(t2) =
        let result = tt.Pow(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.PowT0T(t2) = 
        let result = t1.Expand(t2.Shape).TorchTensor.Pow(t2.TorchTensor)
        t1.CreateLike(result)
        //failwith "PowT0T"

    override t1.PowTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Pow(t2v)
        t1.CreateLike(result)

    override t1.MatMulT2T2(t2) = 
        let result = tt.Bmm(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.Conv1D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _

    override t1.Conv2D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _

    override t1.Conv3D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _

    override t.SumT2Dim0() = failwith "tbd - SumT2Dim0" //(tt.SumT2Dim0)

    override t.NegT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.CreateLike(-tt)

    override t.SumT() =
        match dtype with 
        | DType.Bool -> t.Cast(Int64).SumT()
        | _ ->  t.CreateLike(tt.Sum())

    override t.SignT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Sign())

    override t.FloorT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Floor())

    override t.CeilT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Ceil())

    override t.RoundT() =
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Round())

    override t.AbsT() = 
        match dtype with 
        | DType.Bool -> opNotSupported t.DType
        | _ -> t.CreateLike(tt.Abs ())

    override t.ReluT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->   t.CreateLike(tt.Relu())

    override t.SigmoidT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Sigmoid())

    override t.ExpT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Exp())

    override t.LogT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Log())

    override t.Log10T() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->   t.CreateLike(tt.Log10())

    override t.SqrtT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Sqrt())

    override t.SinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Sin())

    override t.CosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Cos())

    override t.TanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Tan())

    override t.SinhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Sinh())

    override t.CoshT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Cosh())

    override t.TanhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Tanh())

    override t.AsinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Asin())

    override t.AcosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Acos())

    override t.AtanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported t.DType
        | _ ->  t.CreateLike(tt.Atan())

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

    override _.Zero = TorchTensorCPU(from0(conv(zero)).[0L], Shape.scalar, dtype, device) :> _ 
    override _.One =  TorchTensorCPU(from0(conv(one)).[0L], Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:int[]) = TorchTensorCPU(zeros(toTorchShape shape), shape, dtype, device) :> _
    override _.Ones(shape:int[]) = TorchTensorCPU(ones(toTorchShape shape), shape, dtype, device) :> _
    override _.Random(shape:int[]) = TorchTensorCPU(random(toTorchShape shape), shape, dtype, device) :> _
    override _.RandomNormal(shape:int[]) = TorchTensorCPU(randomN(toTorchShape shape), shape, dtype, device) :> _

    override _.Full(shape:int[], value:obj) =
        let t = zeros(toTorchShape shape)
        t.FillInPlace(scalarFromConvValue (conv (valueFromObj value))) |> ignore
        TorchTensorCPU(t, shape, dtype, device) :> _

    override ts.CreateFromFlatArray(values:Array, shape) =
        let values = values :?> 'T[] |> Array.map conv 
        let t = 
            match shape with 
            | [| |] -> from0(values.[0]).[0L]
            | _ -> from (values, toTorchShape shape)
        TorchTensorCPU(t, shape, dtype, device) :> _

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
