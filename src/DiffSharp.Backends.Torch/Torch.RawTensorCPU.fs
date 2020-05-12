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

    let torchScalarShape = [| 1L |]
    let toTorchShape (shape: int[]) = if shape.Length = 0 then torchScalarShape else Array.map int64 shape
    let byteOfBool b = if b then 1uy else 0uy
    let boolOfByte b = (b <> 0uy)

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    let createFromFlatArray createTensor setTensor0 setTensor1 setTensor2 setTensor3 setTensor4 conv (values: 'T[]) shape =
        let t = createTensor(toTorchShape shape)
        match shape with 
        | [| |] -> 
            setTensor0 t (conv values.[0])
        | [| d0 |] ->
            let mutable j = 0
            for i0 in 0 .. d0-1  do  
                setTensor1 t (int64 i0) (conv values.[j])
                j <- j + 1
        | [| d0; d1 |] ->
            let mutable j = 0
            for i0 in 0 .. d0-1  do  
                for i1 in 0 .. d1-1  do  
                    setTensor2 t (int64 i0, int64 i1) (conv values.[j])
                    j <- j + 1
        | [| d0; d1; d2 |] ->
            let mutable j = 0
            for i0 in 0 .. d0-1  do  
                for i1 in 0 .. d1-1  do  
                    for i2 in 0 .. d2-1  do  
                        setTensor3 t (int64 i0, int64 i1, int64 i2) (conv values.[j])
                        j <- j + 1
        | [| d0; d1; d2; d3 |] ->
            let mutable j = 0
            for i0 in 0 .. d0-1  do  
                for i1 in 0 .. d1-1  do  
                    for i2 in 0 .. d2-1  do  
                        for i3 in 0 .. d3-1  do  
                            setTensor4 t (int64 i0, int64 i1, int64 i2, int64 i3) (conv values.[j])
                            j <- j + 1
        | _ -> failwith "Maximum number of dimensions for tensor creation is 4"
        t

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
            if fullBounds.[i, 2] = 1 && len = 1 then 
                res <- res.Squeeze(int64 dim)  // yield len // if len=1 then squeeze this dimension
            else
                let idxs = LongTensor.Arange(int64 start, int64 stop, 1L)
                res <- res.IndexSelect(int64 dim, idxs)  // yield len // if len=1 then squeeze this dimension
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
    
    override t.Expand(newShape) = failwith "TBD - Expand"
        //if shape = newShape then t :> _ else
        //checkCanExpandShape shape newShape
        //let trim = newShape.Length - shape.Length
        //let exp = shapeLength newShape.[0..trim-1]
        //let jshape = newShape.[trim..]
        //let n = shapeLength newShape
        //let result = Array.zeroCreate n 
        //if jshape.Length = 0 then 
        //    // The expansion is everything
        //    for jP = 0 to exp-1 do
        //        result.[jP] <- values.[0]
        //else
        //    for jP = 0 to exp-1 do
        //        let rec loop ibase jbase d = 
        //            let strideD = if (shape.[d] = jshape.[d]) then 1 else 0
        //            if d < jshape.Length-1 then
        //                let mutable iD = 0
        //                for jD = 0 to jshape.[d]-1 do 
        //                    let ibaseD = (ibase+iD)*shape.[d+1]
        //                    let jbaseD = (jbase+jD)*jshape.[d+1]
        //                    loop ibaseD jbaseD (d+1)
        //                    iD <- iD + strideD
        //            else
        //                let mutable iD = 0
        //                // last loop does the actual copy fragments
        //                for jD = 0 to jshape.[d]-1 do 
        //                    result.[jbase+jD] <- values.[ibase+iD]
        //                    iD <- iD + strideD
        //        loop 0 (jP*jshape.[0]) 0
        //t.CreateLike(result, newShape)

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
        let _n, _shape1, _shape2, newShape = Shape.computeStackOp shapes dim
        let result = failwith "tbd" //tts.[0].
        (tensors.[0] :?> TorchTensorCPU).CreateLike(result, newShape)

    override t.UnstackT(dim) = failwith "TBD"
        //checkCanUnstack t.Dim
        //if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        //let shape = t.Shape
        //let n = shape.[dim]
        //let shape1 = shape.[0..dim-1]
        //let shape2 = shape.[dim+1..]
        //let m1 = shapeLength shape1
        //let m2 = shapeLength shape2
        //let unstackedShape = Array.append shape1 shape2
        //let m = m1 * m2
        //let values = t.Values
        //let results = Array.init n (fun _ -> Array.zeroCreate m)
        //for i=0 to (n*m)-1 do
        //    let chunk = i/m2
        //    let i2 = chunk%n
        //    let j2 = (chunk/n)*m2+i%m2
        //    results.[i2].[j2] <- values.[i]
        //results |> Array.map (fun rvalues -> t.CreateLike(rvalues, unstackedShape))

    override t.CatTs(tensors, dim) = failwith "TBD - CatTs"
        //let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorCPU<'T>).Values, t.Shape) |> Array.unzip
        //let n = shapes.Length
        //if n = 0 then invalidArg "tensors" "Expecting at least one tensor"
        //let shape = shapes.[0]
        //if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        //let shape1 = shape.[0..dim-1]
        //let shape2 = shape.[dim+1..]
        //if shapes |> Array.exists (fun shapeOther -> shapeOther.[0..dim-1] <> shape1 || shapeOther.[dim+1..] <> shape2) then
        //    invalidArg "tensors" "Expecting Tensors with similar shapes"
        //let m1 = shapeLength shape1
        //let m2 = shapes |> Array.sumBy (fun shape -> shape.[dim])
        //let m3 = shapeLength shape2
        //let m = m1 * m2 * m3
        //let result = Array.zeroCreate m
        //let outShape = [| yield! shape1; yield m2; yield! shape2 |]
        //let mutable i = 0
        //for j1 = 0 to m1-1 do 
        //    for k = 0 to n-1 do
        //        let d = shapes.[k].[dim]
        //        let b = j1*m3*d
        //        for j2 = 0 to d*m3-1 do
        //            result.[i+j2] <-values.[k].[b+j2]
        //        i <- i + d*m3

        //t.CreateLike(result, outShape)

    override t.SplitT(sizes, dim) = failwith "TBD - SplitT"
        //if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        //let shape = t.Shape
        //if Array.sum sizes <> shape.[dim] then invalidArg "sizes" "the sum of sizes must equal the relevant dimension"
        //let n = sizes.Length
        //let shape1 = shape.[0..dim-1]
        //let shape2 = shape.[dim+1..]
        //let m1 = shapeLength shape1
        //let m3 = shapeLength shape2
        //let values = t.Values
        //let results = Array.init n (fun k -> Array.zeroCreate (m1 * sizes.[k] * m3))
        //let mutable i = 0
        //for j1 = 0 to m1-1 do 
        //    for k = 0 to n-1 do
        //        let d = sizes.[k]
        //        let b = j1*m3*d
        //        for j2 = 0 to d*m3-1 do
        //            results.[k].[b+j2] <-values.[i+j2]
        //        i <- i + d*m3

        //results |> Array.mapi (fun k rvalues -> 
        //    let splitShape = [| yield! shape1; yield sizes.[k]; yield! shape2 |]
        //    t.CreateLike(rvalues, splitShape))

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

    override t.SqueezeT(dim) = failwith "TBD - SqueezeT"
        //let result = Array.copy t.Values
        //t.CreateLike(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) = failwith "TBD - UnsqueezeT"
        //let result = Array.copy t.Values
        //t.CreateLike(result, shapeUnsqueeze dim t.Shape)

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
        failwith "TBD - ViewT"
        //let result = Array.copy t.Values
        //t.CreateLike(result, shape)

    override t.Cast(dtype: DType) =
        if dtype = t.DType then 
            upcast t
        else 
            failwith "TBD - Cast"

    override _.RandomMultinomial(numSamples) =
        failwith "tbd"

    override _.Equals(t2:RawTensor) : bool = 
        shape = t2.Shape && tt.Equal(t2.TorchTensor)

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
        let (struct (_values, indexes)) = tt.TopK(1)
        let t = indexes.[0L]
        Array.init (int t.NumberOfElements) (fun i -> t.[int64 i].DataItem<int64>() |> int32)

    override t.MinIndexT() = 
        let (struct (_values, indexes)) = tt.Neg().TopK(1) // TODO is there a way of doing this withoug Neg()
        let t = indexes.[0L]
        Array.init (int t.NumberOfElements) (fun i -> t.[int64 i].DataItem<int64>() |> int32)

    override t1.AddTT(t2) =
        let result = tt.Add(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.AddTT0(t2) =
        let t2v = t2.TorchTensor.[0L].Item()
        let result = tt.Add(t2v)
        t1.CreateLike(result)

    override t1.AddT2T1(t2) = 
        let result = tt.Add(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        failwith "tbd AddTTSlice" //RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create

    override t1.SubTT(t2) = 
        let result = tt.Sub(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.SubT0T(t2) =
        let t1v = t1.TorchTensor.[0L].Item()
        let result = t1v - t2.TorchTensor
        t1.CreateLike(result)

    override t1.SubTT0(t2) = 
        let t2v = t2.TorchTensor.[0L].Item()
        let result = tt.Sub(t2v)
        t1.CreateLike(result)

    override t1.MulTT(t2) = 
        let result = tt.Mul(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.MulTT0(t2) = 
        let t2v = t2.TorchTensor.[0L].Item()
        let result = tt.Mul(t2v)
        t1.CreateLike(result)

    override t1.DivTT(t2) = 
        let result = tt.Div(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.DivT0T(t2) =
        let t1v = t1.TorchTensor.[0L].Item()
        let result = t1v / t2.TorchTensor
        t1.CreateLike(result)

    override t1.DivTT0(t2) = 
        let t2v = t2.TorchTensor.[0L].Item()
        let result = tt.Div(t2v)
        t1.CreateLike(result)

    override t1.PowTT(t2) =
        let result = tt.Pow(t2.TorchTensor)
        t1.CreateLike(result)

    override t1.PowT0T(t2) = 
        failwith "PowT0T"

    override t1.PowTT0(t2) =
        let t2v = t2.TorchTensor.[0L].Item()
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
        | DType.Bool -> opNotSupported t.DType
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
        from0: 'T2 -> TorchTensor, from: 'T2[] * TorchShape -> TorchTensor,
        zero: 'T, one: 'T,
        zeros: TorchShape -> TorchTensor,
        ones: TorchShape -> TorchTensor,
        random: TorchShape -> TorchTensor,
        randomN: TorchShape -> TorchTensor,
        valueFromObj: obj -> 'T,
        scalarFromConvValue: 'T2 -> Scalar) = 

    inherit RawTensorStatics()

    override _.Zero = TorchTensorCPU(from0(conv(zero)), Shape.scalar, dtype, device) :> _ 
    override _.One =  TorchTensorCPU(from0(conv(one)), Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:int[]) = TorchTensorCPU(zeros(toTorchShape shape), shape, dtype, device) :> _
    override _.Ones(shape:int[]) = TorchTensorCPU(ones(toTorchShape shape), shape, dtype, device) :> _
    override _.Random(shape:int[]) = TorchTensorCPU(random(toTorchShape shape), shape, dtype, device) :> _
    override _.RandomNormal(shape:int[]) = TorchTensorCPU(randomN(toTorchShape shape), shape, dtype, device) :> _

    override _.Full(shape:int[], value:obj) =
        let t = ByteTensor.Zeros(toTorchShape shape)
        t.FillInPlace(scalarFromConvValue (conv (valueFromObj value))) |> ignore
        TorchTensorCPU(t, shape, dtype, device) :> _

    override ts.CreateFromFlatArray(values:Array, shape) =
        let values = values :?> 'T[] |> Array.map conv 
        let t = 
            match shape with 
            | [| |] -> from0(values.[0])
            | _ -> from (values, toTorchShape shape)
        TorchTensorCPU(t, shape, dtype, device) :> _

/// The concrete implementation of RawTensorStatics for Bool CPU data.
type RawTensorFloat32CPUStatics() = 

    inherit TorchStatics<single, single>(DType.Float32, Device.CPU, id, FloatTensor.From, FloatTensor.From, 0.0f, 1.0f, FloatTensor.Zeros, FloatTensor.Ones, ByteTensor.Random, ByteTensor.RandomN, System.Convert.ToSingle, Scalar.op_Implicit)

type RawTensorBoolCPUStatics() = 

    inherit TorchStatics<bool, byte>(DType.Bool, Device.CPU, byteOfBool, ByteTensor.From, ByteTensor.From, false, true, ByteTensor.Zeros, ByteTensor.Ones, ByteTensor.Random, ByteTensor.RandomN, System.Convert.ToBoolean, Scalar.op_Implicit)
