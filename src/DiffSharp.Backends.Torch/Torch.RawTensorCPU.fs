namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open AtenSharp


#nowarn "77" // use of op_Explicit

[<AutoOpen>]
module internal Utils = 
    let opNotSupported (t: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type %A" t)

    let opNotSupported2 (t1: DType) (t2: DType) =
        invalidOp (sprintf "operation not permitted on tensors of type (%A, %A)" t1 t2)

    let torchScalarShape = [| 1L |]
    let toTorchShape (shape: int[]) = if shape.Length = 0 then torchScalarShape else Array.map int64 shape
    let byteOfBool b = if b then 1uy else 0uy

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

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type RawTensorFloat32CPU(tt: FloatTensor, shape: int[]) =
    inherit RawTensor(shape, DType.Float32, CPU, Backend.Torch)

    static let create(tt, shape) : RawTensor = upcast RawTensorFloat32CPU(tt, shape)
    static let createBool(tt, shape) : RawTensor = upcast RawTensorBoolCPU(tt, shape)

    member x.TorchTensor = tt

    override t.GetSlice(fullBounds:int[,]) =
        let newShape = Shape.computeGetSlice fullBounds
        let mutable res = tt
        let mutable dim = 0 
        for i=0 to (fullBounds.GetLength(0) - 1) do
            let start = fullBounds.[i,0]
            let stop = fullBounds.[i,1] + 1

            let len = stop - start
            if fullBounds.[i, 2] = 1 then
                if len > 1 then 
                    res <- res.IndexSelect(dim, LongTensor.ARange(int64 start, int64 stop, 1L))  // yield len // if len=1 then squeeze this dimension
                    dim <- dim + 1
                else
                    res <- res.Select(dim, 0L)  // yield len // if len=1 then squeeze this dimension
            else
                res <- res.IndexSelect(dim, LongTensor.ARange(int64 start, int64 stop, 1L))  // yield len // if len=1 then squeeze this dimension
        create (res, newShape)

    override t.Clone() = RawTensorFloat32CPU(tt.Clone(), shape) :> _ 

    override x.ComputeHash() = hash shape //+ hash values
    
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
        //t.CreateShaped(result, newShape)

    override t.GetItem(indexes) = 
        match indexes with 
        | [| |] -> box tt.[0L]
        | [| i0 |] -> box tt.[int64 i0]
        | [| i0; i1 |] -> box tt.[int64 i0, int64 i1]
        | [| i0; i1; i2 |] -> box tt.[int64 i0, int64 i1, int64 i2]
        | [| i0; i1; i2; i3 |] -> box tt.[int64 i0, int64 i1, int64 i2, int64 i3]
        | _ -> failwith "dim > 4"

    member _.CreateShaped(tt, shape) : RawTensor = upcast RawTensorFloat32CPU(tt, shape)

    override _.StackTs(tensors, dim) =
        let _tts, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorFloat32CPU).TorchTensor, t.Shape) |> Array.unzip
        checkCanStack shapes dim
        let _n, _shape1, _shape2, newShape = Shape.computeStackOp shapes dim
        let result = failwith "tbd" //tts.[0].
        (tensors.[0] :?> RawTensorFloat32CPU).CreateShaped(result, newShape)

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
        //results |> Array.map (fun rvalues -> t.CreateShaped(rvalues, unstackedShape))

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

        //t.CreateShaped(result, outShape)

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
        //    t.CreateShaped(rvalues, splitShape))

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
        //t.CreateShaped(result, newShape)

    override t.SqueezeT(dim) = failwith "TBD - SqueezeT"
        //let result = Array.copy t.Values
        //t.CreateShaped(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) = failwith "TBD - UnsqueezeT"
        //let result = Array.copy t.Values
        //t.CreateShaped(result, shapeUnsqueeze dim t.Shape)

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
        //t.CreateShaped(result, shape)

    override t.Cast(dtype: DType) =
        if dtype = t.DType then 
            upcast t
        else 
            failwith "TBD - Cast"

    override t1.CompareTo(t2) =
        failwith "tbd"

    override t.RandomMultinomial(numSamples) =
        failwith "tbd"

    override t1.Equals(t2:RawTensor) : bool = 
        let t2 = t2 :?> RawTensorFloat32CPU
        tt.Equal(t2.TorchTensor)

    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        failwith "tbd"

    override t.SoftplusT() = failwith "tbd" //RawTensorCPU.SoftplusT(t) |> create

    override t1.LtTT(t2) =
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.LtTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t1.GtTT(t2) =
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.GtTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t1.LeTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.LeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t1.GeTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.GeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t1.EqTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.EqTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t1.NeqTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.NeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> createBool

    override t.MaxIndexT() = failwith "tbd" //RawTensorCPU.MaxIndexT(t)

    override t.MinIndexT() = failwith "tbd" //RawTensorCPU.MinIndexT(t)

    override t1.AddTT(t2) =
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.CAdd(1.0f, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.AddTT0(t2) =
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.Add(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.AddT2T1(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.AddMV(1.0f, 1.0f, tt, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.AddTTSlice(location:int[], t2) = failwith "tbd AddTTSlice" //RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create

    override t1.SubTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.CSub(1.0f, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.SubT0T(t2) = failwith "tbd"
        //let t2 = t2 :?> RawTensorFloat32CPU
        //let result = tt.Sub(t2.TorchTensor.[0L])
        //(result, t1.Shape)  |> create

    override t1.SubTT0(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.Sub(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.MulTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.CMul(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.MulTT0(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.Mul(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.DivTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.CDiv(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.DivT0T(t2) = failwith "tbd" //RawTensorCPU.DivT0T(t1, t2) |> create

    override t1.DivTT0(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.Div(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.PowTT(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.CPow(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.PowT0T(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = t2.TorchTensor.TPow(tt.[0L])
        (result, t1.Shape)  |> create

    override t1.PowTT0(t2) =
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = tt.Pow(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.MatMulT2T2(t2) = 
        let t2 = t2 :?> RawTensorFloat32CPU
        let z = t1.ZeroLike() :?> RawTensorFloat32CPU
        let result = z.TorchTensor.AddBMM(0.0f, 1.0f, t1.TorchTensor, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.Conv1D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _

    override t1.Conv2D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _

    override t1.Conv3D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _

    override t.NegT() = 
        let result = tt.neg()
        (result, t.Shape)  |> create

    override t.SumT() =
        let result = float32 (tt.SumAll())
        let res = new FloatTensor(1L)
        res.[0L] <- result
        (res, t.Shape)  |> create

    override t.SumT2Dim0() = failwith "tbd" //(tt.SumT2Dim0, t.Shape) |> create

    override t.SignT() = (tt.Sign(), t.Shape) |> create

    override t.FloorT() = (tt.Floor(), t.Shape) |> create

    override t.CeilT() = (tt.Ceil(), t.Shape) |> create

    override t.RoundT() = (tt.Round(), t.Shape) |> create

    override t.AbsT() = (tt.Abs(), t.Shape) |> create

    override t.ReluT() =  failwith "TBD Relu" //(tt.Re(), t.Shape) |> create

    override t.SigmoidT() = (tt.Sigmoid(), t.Shape) |> create

    override t.ExpT() = (tt.Exp(), t.Shape) |> create

    override t.LogT() = (tt.Log(), t.Shape) |> create

    override t.Log10T() = (tt.Log10(), t.Shape) |> create

    override t.SqrtT() = (tt.Sqrt(), t.Shape) |> create

    override t.SinT() = (tt.Sin(), t.Shape) |> create

    override t.CosT() = (tt.Cos(), t.Shape) |> create

    override t.TanT() = (tt.Tan(), t.Shape) |> create

    override t.SinhT() = (tt.Sinh(), t.Shape) |> create

    override t.CoshT() = (tt.Cosh(), t.Shape) |> create

    override t.TanhT() = (tt.Tanh(), t.Shape) |> create

    override t.AsinT() = (tt.Asin(), t.Shape) |> create

    override t.AcosT() = (tt.Asin(), t.Shape) |> create

    override t.AtanT() = (tt.Atan(), t.Shape) |> create

/// The concrete implementation of RawTensorStatics for Float32 data.
type RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero =
        let t = new FloatTensor(1L)
        RawTensorFloat32CPU(t, Shape.scalar) :> _ // upcast (RawTensorCPU.Zero() |> RawTensorFloat32CPU)

    override _.One = 
        let t = new FloatTensor(1L)
        t.Fill(1.0f)
        RawTensorFloat32CPU(t, Shape.scalar) :> _

    override _.Zeros(shape:int[]) = 
        let t = new FloatTensor(toTorchShape shape)
        RawTensorFloat32CPU(t, shape) :> _

    override _.Ones(shape:int[]) =
        let t = new FloatTensor(toTorchShape shape)
        t.Fill(1.0f)
        RawTensorFloat32CPU(t, shape) :> _

    override _.Full(shape:int[], value:obj) =
        let t = new FloatTensor(toTorchShape shape)
        t.Fill(System.Convert.ToSingle value)
        RawTensorFloat32CPU(t, shape) :> _

    override _.Random(shape:int[]) =
        failwith "Random tbd" //upcast (RawTensorCPU.Random float32 shape |> RawTensorFloat32CPU)

    override _.RandomNormal(shape:int[]) =
        failwith "RandomNormal tbd" //upcast (RawTensorCPU.RandomNormal float32 shape |> RawTensorFloat32CPU)

    override ts.CreateFromFlatArray(values:Array, shape) =
        let values = values :?> float32[]
        let t = createFromFlatArray 
                    (fun shape -> if shape.Length = 0 then new FloatTensor(1L) else new FloatTensor(shape)) 
                    (fun t v -> t.Fill(v))
                    (fun t (i0) v -> t.[i0] <- v)
                    (fun t (i0,i1) v -> t.[i0,i1] <- v)
                    (fun t (i0,i1,i2) v -> t.[i0,i1,i2] <- v)
                    (fun t (i0,i1,i2,i3) v -> t.[i0,i1,i2,i3] <- v)
                    id
                    values
                    shape
        RawTensorFloat32CPU(t, shape) :> _
    
/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type RawTensorBoolCPU(tt: ByteTensor, shape: int[]) =
    inherit RawTensor(shape, DType.Bool, CPU, Backend.Torch)

    static let create(tt, shape) : RawTensor = upcast RawTensorBoolCPU(tt, shape)
    //static let createBool(tt, shape) : RawTensor = upcast RawTensorBoolCPUCPU(tt, shape)

    member x.TorchTensor = tt
    override t.GetSlice(fullBounds:int[,]) = failwith "tbd - GetSlice"

    override t.Clone() = RawTensorBoolCPU(tt.Clone(), shape) :>_ 

    override x.ComputeHash() = hash shape //+ hash values
    
    override t.Expand(newShape) = failwith "TBD - Expand"

    override t.GetItem(indexes) = 
        match indexes with 
        | [| |] -> box tt.[0L]
        | [| i0 |] -> box tt.[int64 i0]
        | [| i0; i1 |] -> box tt.[int64 i0, int64 i1]
        | [| i0; i1; i2 |] -> box tt.[int64 i0, int64 i1, int64 i2]
        | [| i0; i1; i2; i3 |] -> box tt.[int64 i0, int64 i1, int64 i2, int64 i3]
        | _ -> failwith "dim > 4"

    override t.ToValues() = failwith "tbd"

    member _.CreateShaped(tt, shape) : RawTensor = upcast RawTensorBoolCPU(tt, shape)

    override _.StackTs(tensors, dim) =
        let _tts, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorBoolCPU).TorchTensor, t.Shape) |> Array.unzip
        checkCanStack shapes dim
        let _n, _shape1, _shape2, newShape = Shape.computeStackOp shapes dim
        let result = failwith "tbd" //tts.[0].
        (tensors.[0] :?> RawTensorBoolCPU).CreateShaped(result, newShape)

    override t.UnstackT(dim) = failwith "TBD"

    override t.CatTs(tensors, dim) = failwith "TBD - CatTs"

    override t.SplitT(sizes, dim) = failwith "TBD - SplitT"

    override t.TransposeT2() = failwith "TBD - TransposeT2"

    override t.SqueezeT(dim) = failwith "TBD - SqueezeT"

    override t.UnsqueezeT(dim) = failwith "TBD - UnsqueezeT"

    override t.FlipT(dims:int[]) = failwith "TBD - FlipT"

    override t.DilateT(dilations:int[]) = failwith "TBD - DilateT"

    override t.UndilateT(dilations:int[]) = failwith "TBD - UndilateT"

    override t.ViewT(shape:int[]) = failwith "TBD - ViewT"

    override t.Cast(dtype: DType) = failwith "TBD - Cast"

    override t1.CompareTo(t2) = failwith "tbd" 

    override t.RandomMultinomial(numSamples) = failwith "tbd"

    override t1.Equals(t2:RawTensor) : bool = 
        let t2 = t2 :?> RawTensorBoolCPU
        tt.Equal(t2.TorchTensor)

    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = failwith "tbd"

    override t.SoftplusT() = failwith "tbd" 

    override t1.LtTT(t2) =
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.LtTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.GtTT(t2) =
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.GtTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.LeTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.LeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.GeTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.GeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.EqTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.EqTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.NeqTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.NeTensor(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t.MaxIndexT() = failwith "tbd" 

    override t.MinIndexT() = failwith "tbd" 

    override t1.AddTT(t2) =
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.CAdd(1uy, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.AddTT0(t2) =
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.Add(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.AddT2T1(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.AddMV(1uy, 1uy, tt, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.AddTTSlice(location:int[], t2) = failwith "tbd AddTTSlice"

    override t1.SubTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.CSub(1uy, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.SubT0T(t2) = failwith "tbd"
        //let t2 = t2 :?> RawTensorBoolCPU
        //let result = tt.Sub(t2.TorchTensor.[0L])
        //(result, t1.Shape)  |> create

    override t1.SubTT0(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.Sub(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.MulTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.CMul(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.MulTT0(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.Mul(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.DivTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.CDiv(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.DivT0T(t2) = failwith "tbd" 

    override t1.DivTT0(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.Div(t2.TorchTensor.[0L])
        (result, t1.Shape)  |> create

    override t1.PowTT(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let result = tt.CPow(t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.PowT0T(t2) = failwith "tbd" 

    override t1.PowTT0(t2) = failwith "tbd" 

    override t1.MatMulT2T2(t2) = 
        let t2 = t2 :?> RawTensorBoolCPU
        let z = t1.ZeroLike() :?> RawTensorBoolCPU
        let result = z.TorchTensor.AddBMM(0uy, 1uy, t1.TorchTensor, t2.TorchTensor)
        (result, t1.Shape)  |> create

    override t1.Conv1D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _

    override t1.Conv2D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _

    override t1.Conv3D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _

    override t.NegT() = failwith "tbd" 

    override t.SumT() =
        let result = byte (tt.SumAll())
        let res = new ByteTensor(1L)
        res.[0L] <- result
        (res, t.Shape)  |> create

    override t.SumT2Dim0() = failwith "tbd" //(tt.SumT2Dim0, t.Shape) |> create

    override t.SignT() = (tt.Sign(), t.Shape) |> create

    override t.FloorT() = opNotSupported t.DType

    override t.CeilT() = opNotSupported t.DType

    override t.RoundT() = opNotSupported t.DType

    override t.AbsT() = opNotSupported t.DType

    override t.ReluT() =  opNotSupported t.DType

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

/// The concrete implementation of RawTensorStatics for Float32 data.
type RawTensorBoolCPUStatics() = 

    inherit RawTensorStatics()

    override _.Zero =
        let t = new ByteTensor(1L)
        RawTensorBoolCPU(t, Shape.scalar) :> _ 

    override _.One = 
        let t = new ByteTensor(1L)
        t.Fill(1uy)
        RawTensorBoolCPU(t, Shape.scalar) :> _

    override _.Zeros(shape:int[]) = 
        let t = new ByteTensor(toTorchShape shape)
        RawTensorBoolCPU(t, shape) :> _

    override ts.Ones(shape:int[]) =
        let t = new ByteTensor(toTorchShape shape)
        t.Fill(1uy)
        RawTensorBoolCPU(t, shape) :> _

    override _.Full(shape:int[], value:obj) =
        let t = new ByteTensor(toTorchShape shape)
        t.Fill(byteOfBool (System.Convert.ToBoolean value))
        RawTensorBoolCPU(t, shape) :> _

    override _.Random(shape:int[]) =
        failwith "Random tbd" //upcast (RawTensorCPU.Random float32 shape |> RawTensorBoolCPU)

    override _.RandomNormal(shape:int[]) =
        failwith "RandomNormal tbd" //upcast (RawTensorCPU.RandomNormal float32 shape |> RawTensorBoolCPU)

    override ts.CreateFromFlatArray(values:Array, shape) =
        let values = values :?> bool[]
        let t = createFromFlatArray 
                    (fun shape -> if shape.Length = 0 then new ByteTensor(1L) else new ByteTensor(shape)) 
                    (fun t v -> t.Fill(v))
                    (fun t (i0) v -> t.[i0] <- v)
                    (fun t (i0,i1) v -> t.[i0,i1] <- v)
                    (fun t (i0,i1,i2) v -> t.[i0,i1,i2] <- v)
                    (fun t (i0,i1,i2,i3) v -> t.[i0,i1,i2,i3] <- v)
                    byteOfBool
                    values
                    shape
        RawTensorBoolCPU(t, shape) :> _
