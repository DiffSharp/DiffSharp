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

/// This is the base class for all RawTensorXyzCPU tuypes.
/// All type-independent operations are implemented directly on this class. 
type RawTensorSingle(aten: FloatTensor, shape: int[]) =
    inherit RawTensor(shape, DType.Float32, CPU, Backend.Torch)
    do failwith "tbd"

    static let create(aten, shape) : RawTensor = upcast RawTensorSingle(aten, shape)
    //static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

    override t.GetSlice(fullBounds:int[,]) = failwith "tbd"
    //    let shape =
    //        [|for i=0 to (fullBounds.GetLength(0) - 1) do
    //            let len = fullBounds.[i,1] - fullBounds.[i,0] + 1
    //            if fullBounds.[i, 2] = 1 then
    //                if len > 1 then yield len // if len=1 then squeeze this dimension
    //            else
    //                yield len|]
    //    // printfn "rshape\n%A" shape
    //    let array = Array.zeroCreate (shapeLength shape)
    //    let mutable arrayi = 0
    //    let rec slice (fullBounds:int[,]) externalCoords =
    //        if fullBounds.GetLength(0) = 1 then
    //            for i=fullBounds.[0,0] to fullBounds.[0,1] do
    //                // printfn "inner %A" i
    //                let globalCoords = Array.append externalCoords [|i|]
    //                array.[arrayi] <- t.[globalCoords]
    //                arrayi <- arrayi + 1
    //        else
    //            for i=fullBounds.[0,0] to fullBounds.[0,1] do
    //                // printfn "outer %A" i
    //                slice fullBounds.[1..,*] (Array.append externalCoords [|i|])
    //    slice fullBounds [||]
    //    t.CreateShaped(array, shape)

    override t.Clone() = failwith "tbd" //t.CreateShaped(Array.copy t.Values, Array.copy t.Shape)

    //abstract member CreateShaped: values: 'T[] * shape: int[] -> RawTensor

    override t.GetString() = ""
        //// sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        //let printVal (x:obj) = 
        //   match x with 
        //   | :? single as v -> sprintf "%f" v
        //   | :? double as v -> sprintf "%f" v
        //   | :? int8 as v -> sprintf "%d" v
        //   | :? int16 as v -> sprintf "%d" v
        //   | :? int32 as v -> sprintf "%d" v
        //   | :? int64 as v -> sprintf "%d" v
        //   | :? bool as v -> if v then "true" else "false"
        //   | _ -> sprintf "%A" x

        //match t.Dim with
        //| 0 -> printVal t.Values.[0]
        //| _ ->
        //    let sb = System.Text.StringBuilder()
        //    let rec print (shape:int[]) externalCoords = 
        //        if shape.Length = 1 then
        //            sb.Append("[") |> ignore
        //            let mutable prefix = ""
        //            for i=0 to shape.[0]-1 do
        //                let globalCoords = Array.append externalCoords [|i|]
        //                sb.Append(prefix) |> ignore
        //                sb.Append(printVal (t.[globalCoords])) |> ignore
        //                prefix <- ", "
        //            sb.Append("]") |> ignore
        //        else
        //            sb.Append("[") |> ignore
        //            let mutable prefix = ""
        //            let prefix2 = sprintf ", %s%s" (String.replicate (max 1 (shape.Length-1)) "\n") (String.replicate (externalCoords.Length+1) " ")
        //            for i=0 to shape.[0]-1 do
        //                sb.Append(prefix) |> ignore
        //                print shape.[1..] (Array.append externalCoords [|i|])
        //                prefix <- prefix2
        //            sb.Append("]") |> ignore
        //    print t.Shape [||]
        //    sb.ToString()

    override x.ComputeHash() = hash shape //+ hash values
    
    override t.Expand(newShape) = failwith "TBD"
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

    override t.ToValues() = failwith "tbd"
        //match t.Dim with
        //| 0 -> box values.[0]
        //| 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        //| 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        //| 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        //| 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        //| _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override _.StackTs(tensors, dim) = failwith "TBD"
        //let values, shapes = tensors |> Array.map (fun t -> (t :?> RawTensorCPU<'T>).Values, t.Shape) |> Array.unzip
        //checkCanStack shapes
        //let shape = shapes.[0]
        //if dim < 0 || dim > shape.Length then invalidArg "dim" "invalid dimension"
        //let n = tensors |> Array.length
        //let shape1 = shape.[0..dim-1]
        //let shape2 = shape.[dim..]
        //let m1 = shapeLength shape1
        //let m2 = shapeLength shape2
        //let m = m1 * m2
        //let result = Array.zeroCreate (n * m)
        //for i=0 to (n*m)-1 do
        //    let chunk = i/m2
        //    let i2 = chunk%n
        //    let j2 = (chunk/n)*m2+i%m2
        //    result.[i] <-values.[i2].[j2]

        //let outShape = [| yield! shape1; yield n; yield! shape2 |]
        //(tensors.[0] :?> RawTensorCPU<'T>).CreateShaped(result, outShape)

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

    override t.CatTs(tensors, dim) = failwith "TBD"
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

    override t.SplitT(sizes, dim) = failwith "TBD"
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

    override t.TransposeT2() = failwith "TBD"
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

    override t.SqueezeT(dim) = failwith "TBD"
        //let result = Array.copy t.Values
        //t.CreateShaped(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) = failwith "TBD"
        //let result = Array.copy t.Values
        //t.CreateShaped(result, shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) = failwith "TBD"
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

    override t.DilateT(dilations:int[]) = failwith "TBD"
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

    override t.UndilateT(dilations:int[]) = failwith "TBD"
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

    override t.ViewT(shape:int[]) = failwith "TBD"
        //checkCanView t.Shape shape
        //let result = Array.copy t.Values
        //t.CreateShaped(result, shape)

    override t.Cast(dtype: DType) = failwith "TBD"
        //if dtype = t.DType then 
        //    upcast t
        //else 
        //    RawTensor.Create(t.ToValues(), dtype=dtype, backend=t.Backend, device=t.Device)

    override t1.CompareTo(t2) = failwith "tbd" //RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorSingle))
    //override t.CreateShaped(values, shape) = upcast RawTensorSingle(values, shape)
    override t.RandomMultinomial(numSamples) = failwith "tbd" //RawTensorCPU.RandomMultinomial float32 (t, numSamples)|> create
    override t1.Equals(t2:RawTensor) : bool = failwith "tbd" //RawTensorCPU.Equals(t1, t2)
    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = failwith "tbd" //RawTensorCPU.AllClose(t1, t2, float32 relativeTolerance, float32 absoluteTolerance)
    override t.IsInfT() = failwith "tbd" //RawTensorCPU.IsInfT(System.Single.IsInfinity, t) |> createBool
    override t.IsNaNT() = failwith "tbd" //RawTensorCPU.IsNaNT(System.Single.IsNaN, t) |> createBool
    override t.SoftplusT() = failwith "tbd" //RawTensorCPU.SoftplusT(t) |> create
    override t1.LtTT(t2) = failwith "tbd" //RawTensorCPU.LtTT(t1, t2) |> createBool
    override t1.GtTT(t2) = failwith "tbd" //RawTensorCPU.GtTT(t1, t2) |> createBool
    override t1.LeTT(t2) = failwith "tbd" //RawTensorCPU.LeTT(t1, t2) |> createBool
    override t1.GeTT(t2) = failwith "tbd" //RawTensorCPU.GeTT(t1, t2) |> createBool
    override t.MaxIndexT() = failwith "tbd" //RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = failwith "tbd" //RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = failwith "tbd" //RawTensorCPU.AddTT(t1, t2) |> create
    override t1.AddTT0(t2) = failwith "tbd" //RawTensorCPU.AddTT0(t1, t2) |> create
    override t1.AddT2T1(t2) = failwith "tbd" //RawTensorCPU.AddT2T1(t1, t2) |> create
    override t1.AddTTSlice(location:int[], t2) = failwith "tbd" //RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
    override t1.SubTT(t2) = failwith "tbd" //RawTensorCPU.SubTT(t1, t2) |> create
    override t1.SubT0T(t2) = failwith "tbd" //RawTensorCPU.SubT0T(t1, t2) |> create
    override t1.SubTT0(t2) = failwith "tbd" //RawTensorCPU.SubTT0(t1, t2) |> create
    override t1.MulTT(t2) = failwith "tbd" //RawTensorCPU.MulTT(t1, t2) |> create
    override t1.MulTT0(t2) = failwith "tbd" //RawTensorCPU.MulTT0(t1, t2) |> create
    override t1.DivTT(t2) = failwith "tbd" //RawTensorCPU.DivTT(t1, t2) |> create
    override t1.DivT0T(t2) = failwith "tbd" //RawTensorCPU.DivT0T(t1, t2) |> create
    override t1.DivTT0(t2) = failwith "tbd" //RawTensorCPU.DivTT0(t1, t2) |> create
    override t1.PowTT(t2) = failwith "tbd" //RawTensorCPU.PowTT(t1, t2) |> create
    override t1.PowT0T(t2) = failwith "tbd" //RawTensorCPU.PowT0T(t1, t2) |> create
    override t1.PowTT0(t2) = failwith "tbd" //RawTensorCPU.PowTT0(t1, t2) |> create
    override t1.MatMulT2T2(t2) = failwith "tbd" //RawTensorCPU.MatMulTT(t1, t2) |> create
    override t1.Conv1D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
    override t1.Conv2D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
    override t1.Conv3D(t2, stride, padding) = failwith "tbd" //RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
    override t.NegT() = failwith "tbd" //RawTensorCPU.NegT(t) |> create
    override t.SumT() = failwith "tbd" //RawTensorCPU.SumT(t) |> create
    override t.SumT2Dim0() = failwith "tbd" //RawTensorCPU.SumT2Dim0(t) |> create
    override t.SignT() = failwith "tbd" //RawTensorCPU.SignT float32 t |> create
    override t.FloorT() = failwith "tbd" //RawTensorCPU.FloorT(t) |> create
    override t.CeilT() = failwith "tbd" //RawTensorCPU.CeilT(t) |> create
    override t.RoundT() = failwith "tbd" //RawTensorCPU.RoundT(t) |> create
    override t.AbsT() = failwith "tbd" //RawTensorCPU.AbsT(t) |> create
    override t.ReluT() = failwith "tbd" //RawTensorCPU.ReluT(t) |> create
    override t.SigmoidT() = failwith "tbd" //RawTensorCPU.SigmoidT(t) |> create
    override t.ExpT() = failwith "tbd" //RawTensorCPU.ExpT(t) |> create
    override t.LogT() = failwith "tbd" //RawTensorCPU.LogT(t) |> create
    override t.Log10T() = failwith "tbd" //RawTensorCPU.Log10T(t) |> create
    override t.SqrtT() = failwith "tbd" //RawTensorCPU.SqrtT(t) |> create
    override t.SinT() = failwith "tbd" //RawTensorCPU.SinT(t) |> create
    override t.CosT() = failwith "tbd" //RawTensorCPU.CosT(t) |> create
    override t.TanT() = failwith "tbd" //RawTensorCPU.TanT(t) |> create
    override t.SinhT() = failwith "tbd" //RawTensorCPU.SinhT(t) |> create
    override t.CoshT() = failwith "tbd" //RawTensorCPU.CoshT(t) |> create
    override t.TanhT() = failwith "tbd" //RawTensorCPU.TanhT(t) |> create
    override t.AsinT() = failwith "tbd" //RawTensorCPU.AsinT(t) |> create
    override t.AcosT() = failwith "tbd" //RawTensorCPU.AcosT(t) |> create
    override t.AtanT() = failwith "tbd" //RawTensorCPU.AtanT(t) |> create

/// The concrete implementation of RawTensorStatics for Float32 data.
type RawTensorSingleStatics() = 

    inherit RawTensorStatics()

    override _.Zero = RawTensorSingle(new FloatTensor(), Shape.scalar) :> _ // upcast (RawTensorCPU.Zero() |> RawTensorSingle)
    override _.One = failwith "tbd" //upcast (RawTensorCPU.One() |> RawTensorSingle)
    override _.Zeros(shape:int[]) = failwith "tbd" //upcast (RawTensorCPU.Zeros(shape) |> RawTensorSingle)
    override _.Ones(shape:int[]) = failwith "tbd" //upcast (RawTensorCPU.Ones(shape) |> RawTensorSingle)
    override _.Full(shape:int[], value:obj) = failwith "tbd" //upcast (RawTensorCPU.Full (shape, System.Convert.ToSingle value) |> RawTensorSingle)
    override _.Random(shape:int[]) = failwith "tbd" //upcast (RawTensorCPU.Random float32 shape |> RawTensorSingle)
    override _.RandomNormal(shape:int[]) = failwith "tbd" //upcast (RawTensorCPU.RandomNormal float32 shape |> RawTensorSingle)
    override _.CreateFromFlatArray(values:Array, shape) = failwith "tbd" //upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorSingle)

//type RawTensorFloat64CPU(values: double[], shape:int[]) =
//    inherit RawTensorCPU<double>(values, shape, Float64)

//    static let create(values, shape) : RawTensor = failwith "tbd" //upcast RawTensorFloat64CPU(values, shape)
//    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
//    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial double (t, numSamples)|> create
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) = RawTensorCPU.AllClose(t1, t2, relativeTolerance, absoluteTolerance)
//    override t.IsInfT() = RawTensorCPU.IsInfT(System.Double.IsInfinity, t) |> createBool
//    override t.IsNaNT() = RawTensorCPU.IsNaNT(System.Double.IsNaN, t) |> createBool
//    override t.SoftplusT() = RawTensorCPU.SoftplusT(t) |> create
//    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
//    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
//    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
//    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
//    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
//    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
//    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
//    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
//    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
//    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
//    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
//    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
//    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
//    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
//    override t1.PowTT(t2) = RawTensorCPU.PowTT(t1, t2) |> create
//    override t1.PowT0T(t2) = RawTensorCPU.PowT0T(t1, t2) |> create
//    override t1.PowTT0(t2) = RawTensorCPU.PowTT0(t1, t2) |> create
//    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
//    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D (t1, t2, stride, padding) :> _
//    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
//    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
//    override t.NegT() = RawTensorCPU.NegT(t) |> create
//    override t.SumT() = RawTensorCPU.SumT(t) |> create
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
//    override t.SignT() = RawTensorCPU.SignT double t |> create
//    override t.FloorT() = RawTensorCPU.FloorT(t) |> create
//    override t.CeilT() = RawTensorCPU.CeilT(t) |> create
//    override t.RoundT() = RawTensorCPU.RoundT(t) |> create
//    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
//    override t.ReluT() = RawTensorCPU.ReluT(t) |> create
//    override t.SigmoidT() = RawTensorCPU.SigmoidT(t) |> create
//    override t.ExpT() = RawTensorCPU.ExpT(t) |> create
//    override t.LogT() = RawTensorCPU.LogT(t) |> create
//    override t.Log10T() = RawTensorCPU.Log10T(t) |> create
//    override t.SqrtT() = RawTensorCPU.SqrtT(t) |> create
//    override t.SinT() = RawTensorCPU.SinT(t) |> create
//    override t.CosT() = RawTensorCPU.CosT(t) |> create
//    override t.TanT() = RawTensorCPU.TanT(t) |> create
//    override t.SinhT() = RawTensorCPU.SinhT(t) |> create
//    override t.CoshT() = RawTensorCPU.CoshT(t) |> create
//    override t.TanhT() = RawTensorCPU.TanhT(t) |> create
//    override t.AsinT() = RawTensorCPU.AsinT(t) |> create
//    override t.AcosT() = RawTensorCPU.AcosT(t) |> create
//    override t.AtanT() = RawTensorCPU.AtanT(t) |> create

//type RawTensorFloat64CPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorFloat64CPU)
//    override _.One = upcast (RawTensorCPU.One() |> RawTensorFloat64CPU)
//    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorFloat64CPU)
//    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorFloat64CPU)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToDouble value) |> RawTensorFloat64CPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random double shape |> RawTensorFloat64CPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal double shape |> RawTensorFloat64CPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorFloat64CPU)

//type RawTensorInt8CPU(values: int8[], shape:int[]) =
//    inherit RawTensorCPU<int8>(values, shape, Int8)

//    static let create(values, shape) : RawTensor = upcast RawTensorInt8CPU(values, shape)
//    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt8CPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorInt8CPU(values, shape)
//    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int8 (t, numSamples)|> create
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
//    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
//    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
//    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
//    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
//    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
//    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
//    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
//    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
//    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
//    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
//    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
//    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
//    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
//    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
//    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
//    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
//    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
//    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
//    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
//    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
//    override t.NegT() = RawTensorCPU.NegT(t) |> create
//    override t.SumT() = RawTensorCPU.SumT(t) |> create
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
//    override t.SignT() = RawTensorCPU.SignT int8 t |> create
//    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
//    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

//    override t.SoftplusT() = opNotSupported t.DType
//    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t.FloorT() = opNotSupported t.DType
//    override t.CeilT() = opNotSupported t.DType
//    override t.RoundT() = opNotSupported t.DType
//    override t.SigmoidT() = opNotSupported t.DType
//    override t.ExpT() = opNotSupported t.DType
//    override t.LogT() = opNotSupported t.DType
//    override t.Log10T() = opNotSupported t.DType
//    override t.SqrtT() = opNotSupported t.DType
//    override t.SinT() = opNotSupported t.DType
//    override t.CosT() = opNotSupported t.DType
//    override t.TanT() = opNotSupported t.DType
//    override t.SinhT() = opNotSupported t.DType
//    override t.CoshT() = opNotSupported t.DType
//    override t.TanhT() = opNotSupported t.DType
//    override t.AsinT() = opNotSupported t.DType
//    override t.AcosT() = opNotSupported t.DType
//    override t.AtanT() = opNotSupported t.DType

//type RawTensorInt8CPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt8CPU)
//    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt8CPU)
//    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt8CPU)
//    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt8CPU)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToSByte value) |> RawTensorInt8CPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int8 shape |> RawTensorInt8CPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int8 shape |> RawTensorInt8CPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt8CPU)

//type RawTensorInt16CPU(values: int16[], shape:int[]) =
//    inherit RawTensorCPU<int16>(values, shape, Int16)

//    static let create(values, shape) : RawTensor = upcast RawTensorInt16CPU(values, shape)
//    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt16CPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorInt16CPU(values, shape)
//    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int16 (t, numSamples)|> create
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
//    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
//    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
//    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
//    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
//    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
//    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
//    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
//    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
//    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
//    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
//    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
//    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
//    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
//    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
//    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
//    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
//    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
//    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
//    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
//    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
//    override t.NegT() = RawTensorCPU.NegT(t) |> create
//    override t.SumT() = RawTensorCPU.SumT(t) |> create
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
//    override t.SignT() = RawTensorCPU.SignT int16 t |> create
//    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
//    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

//    override t.SoftplusT() = opNotSupported t.DType
//    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t.FloorT() = opNotSupported t.DType
//    override t.CeilT() = opNotSupported t.DType
//    override t.RoundT() = opNotSupported t.DType
//    override t.SigmoidT() = opNotSupported t.DType
//    override t.ExpT() = opNotSupported t.DType
//    override t.LogT() = opNotSupported t.DType
//    override t.Log10T() = opNotSupported t.DType
//    override t.SqrtT() = opNotSupported t.DType
//    override t.SinT() = opNotSupported t.DType
//    override t.CosT() = opNotSupported t.DType
//    override t.TanT() = opNotSupported t.DType
//    override t.SinhT() = opNotSupported t.DType
//    override t.CoshT() = opNotSupported t.DType
//    override t.TanhT() = opNotSupported t.DType
//    override t.AsinT() = opNotSupported t.DType
//    override t.AcosT() = opNotSupported t.DType
//    override t.AtanT() = opNotSupported t.DType

//type RawTensorInt16CPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt16CPU)
//    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt16CPU)
//    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt16CPU)
//    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt16CPU)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt16 value) |> RawTensorInt16CPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int16 shape |> RawTensorInt16CPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int16 shape |> RawTensorInt16CPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt16CPU)

//type RawTensorInt32CPU(values: int32[], shape:int[]) =
//    inherit RawTensorCPU<int32>(values, shape, Int32)

//    static let create(values, shape) : RawTensor = upcast RawTensorInt32CPU(values, shape)
//    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt32CPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorInt32CPU(values, shape)
//    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int32 (t, numSamples)|> create
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
//    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
//    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
//    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
//    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
//    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
//    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
//    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
//    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
//    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
//    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
//    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
//    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
//    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
//    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
//    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
//    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
//    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
//    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
//    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
//    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
//    override t.NegT() = RawTensorCPU.NegT(t) |> create
//    override t.SumT() = RawTensorCPU.SumT(t) |> create
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
//    override t.SignT() = RawTensorCPU.SignT int32 t |> create
//    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
//    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

//    override t.SoftplusT() = opNotSupported t.DType
//    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t.FloorT() = opNotSupported t.DType
//    override t.CeilT() = opNotSupported t.DType
//    override t.RoundT() = opNotSupported t.DType
//    override t.SigmoidT() = opNotSupported t.DType
//    override t.ExpT() = opNotSupported t.DType
//    override t.LogT() = opNotSupported t.DType
//    override t.Log10T() = opNotSupported t.DType
//    override t.SqrtT() = opNotSupported t.DType
//    override t.SinT() = opNotSupported t.DType
//    override t.CosT() = opNotSupported t.DType
//    override t.TanT() = opNotSupported t.DType
//    override t.SinhT() = opNotSupported t.DType
//    override t.CoshT() = opNotSupported t.DType
//    override t.TanhT() = opNotSupported t.DType
//    override t.AsinT() = opNotSupported t.DType
//    override t.AcosT() = opNotSupported t.DType
//    override t.AtanT() = opNotSupported t.DType

//type RawTensorInt32CPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt32CPU)
//    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt32CPU)
//    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt32CPU)
//    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt32CPU)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt32 value) |> RawTensorInt32CPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int32 shape |> RawTensorInt32CPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int32 shape |> RawTensorInt32CPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt32CPU)
                
//type RawTensorInt64CPU(values: int64[], shape:int[]) =
//    inherit RawTensorCPU<int64>(values, shape, Int64)

//    static let create(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
//    static let createBool(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)

//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorInt64CPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorInt64CPU(values, shape)
//    override t.RandomMultinomial(numSamples) = RawTensorCPU.RandomMultinomial int64 (t, numSamples)|> create
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
//    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> createBool
//    override t.IsNaNT() = RawTensorCPU.IsNaNT((fun _ -> false), t) |> createBool
//    override t1.LtTT(t2) = RawTensorCPU.LtTT(t1, t2) |> createBool
//    override t1.GtTT(t2) = RawTensorCPU.GtTT(t1, t2) |> createBool
//    override t1.LeTT(t2) = RawTensorCPU.LeTT(t1, t2) |> createBool
//    override t1.GeTT(t2) = RawTensorCPU.GeTT(t1, t2) |> createBool
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorCPU.AddTT(t1, t2) |> create
//    override t1.AddTT0(t2) = RawTensorCPU.AddTT0(t1, t2) |> create
//    override t1.AddT2T1(t2) = RawTensorCPU.AddT2T1(t1, t2) |> create
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((+), t1, location, t2) |> create
//    override t1.SubTT(t2) = RawTensorCPU.SubTT(t1, t2) |> create
//    override t1.SubT0T(t2) = RawTensorCPU.SubT0T(t1, t2) |> create
//    override t1.SubTT0(t2) = RawTensorCPU.SubTT0(t1, t2) |> create
//    override t1.MulTT(t2) = RawTensorCPU.MulTT(t1, t2) |> create
//    override t1.MulTT0(t2) = RawTensorCPU.MulTT0(t1, t2) |> create
//    override t1.DivTT(t2) = RawTensorCPU.DivTT(t1, t2) |> create
//    override t1.DivT0T(t2) = RawTensorCPU.DivT0T(t1, t2) |> create
//    override t1.DivTT0(t2) = RawTensorCPU.DivTT0(t1, t2) |> create
//    override t1.MatMulTT(t2) = RawTensorCPU.MatMulTT(t1, t2) |> create
//    override t1.Conv1D(t2, stride, padding) = RawTensorCPU.Conv1D(t1, t2, stride, padding) :> _
//    override t1.Conv2D(t2, stride, padding) = RawTensorCPU.Conv2D (t1, t2, stride, padding) :> _
//    override t1.Conv3D(t2, stride, padding) = RawTensorCPU.Conv3D (t1, t2, stride, padding) :> _
//    override t.NegT() = RawTensorCPU.NegT(t) |> create
//    override t.SumT() = RawTensorCPU.SumT(t) |> create
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t) |> create
//    override t.SignT() = RawTensorCPU.SignT int64 t |> create
//    override t.AbsT() = RawTensorCPU.AbsT(t) |> create
//    override t.ReluT() = RawTensorCPU.ReluT(t) |> create

//    override t.SoftplusT() = opNotSupported t.DType
//    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t.FloorT() = opNotSupported t.DType
//    override t.CeilT() = opNotSupported t.DType
//    override t.RoundT() = opNotSupported t.DType
//    override t.SigmoidT() = opNotSupported t.DType
//    override t.ExpT() = opNotSupported t.DType
//    override t.LogT() = opNotSupported t.DType
//    override t.Log10T() = opNotSupported t.DType
//    override t.SqrtT() = opNotSupported t.DType
//    override t.SinT() = opNotSupported t.DType
//    override t.CosT() = opNotSupported t.DType
//    override t.TanT() = opNotSupported t.DType
//    override t.SinhT() = opNotSupported t.DType
//    override t.CoshT() = opNotSupported t.DType
//    override t.TanhT() = opNotSupported t.DType
//    override t.AsinT() = opNotSupported t.DType
//    override t.AcosT() = opNotSupported t.DType
//    override t.AtanT() = opNotSupported t.DType

//type RawTensorInt64CPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast (RawTensorCPU.Zero() |> RawTensorInt64CPU)
//    override _.One = upcast (RawTensorCPU.One() |> RawTensorInt64CPU)
//    override _.Zeros(shape:int[]) = upcast (RawTensorCPU.Zeros(shape) |> RawTensorInt64CPU)
//    override _.Ones(shape:int[]) = upcast (RawTensorCPU.Ones(shape) |> RawTensorInt64CPU)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToInt64 value) |> RawTensorInt64CPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random int64 shape |> RawTensorInt64CPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.RandomNormal int64 shape |> RawTensorInt64CPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorInt64CPU)

//type RawTensorBoolCPU(values: bool[], shape:int[]) =
//    inherit RawTensorCPU<bool>(values, shape, Bool)

//    static let create(values, shape) : RawTensor = upcast RawTensorBoolCPU(values, shape)
//    static let create64(values, shape) : RawTensor = upcast RawTensorInt64CPU(values, shape)
       
//    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorBoolCPU))
//    override t.CreateShaped(values, shape) = upcast RawTensorBoolCPU(values, shape)
//    override t.RandomMultinomial(_numSamples) = opNotSupported t.DType
//    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
//    override t1.AllClose(t2:RawTensor, _relativeTolerance, _absoluteTolerance) = RawTensorCPU.Equals(t1, t2)
//    override t1.LtTT(t2) = RawTensorBoolCPU(Array.map2 (<) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t1.GtTT(t2) = RawTensorBoolCPU(Array.map2 (>) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t1.LeTT(t2) = RawTensorBoolCPU(Array.map2 (<=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t1.GeTT(t2) = RawTensorBoolCPU(Array.map2 (>=) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
//    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
//    override t1.AddTT(t2) = RawTensorBoolCPU(Array.map2 (||) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t1.AddTT0(t2) = t1.AddTT(t2.Expand(t1.Shape))
//    override t1.AddT2T1(t2) = t1.AddTT(t2.Expand(t1.Shape))
//    override t1.AddTTSlice(location:int[], t2) = RawTensorCPU.AddTTSlice((||), t1, location, t2) |> create
//    override t1.MulTT(t2) = RawTensorBoolCPU(Array.map2 (&&) t1.Values (t2 :?> RawTensorCPU<bool>).Values, t1.Shape) :> _
//    override t1.MulTT0(t2) = t1.MulTT(t2.Expand(t1.Shape))
//    override t.SumT() = RawTensorCPU.SumT(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
//    override t.SumT2Dim0() = RawTensorCPU.SumT2Dim0(t.Cast(Int64) :?> RawTensorCPU<int64>) |> create64
//    override t.SignT() = t :> _
//    override t.IsInfT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create
//    override t.IsNaNT() = RawTensorCPU.IsInfT((fun _ -> false), t) |> create

//    override t1.SubTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.SubT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.SubTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.DivTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.DivT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.DivTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.MatMulTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.Conv1D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
//    override t1.Conv2D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
//    override t1.Conv3D(t2, _stride, _padding) = opNotSupported2 t1.DType t2.DType
//    override t.NegT() = opNotSupported t.DType
//    override t.AbsT() = opNotSupported t.DType
//    override t.ReluT() = opNotSupported t.DType
//    override t.SoftplusT() = opNotSupported t.DType
//    override t1.PowTT(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowT0T(t2) = opNotSupported2 t1.DType t2.DType
//    override t1.PowTT0(t2) = opNotSupported2 t1.DType t2.DType
//    override t.FloorT() = opNotSupported t.DType
//    override t.CeilT() = opNotSupported t.DType
//    override t.RoundT() = opNotSupported t.DType
//    override t.SigmoidT() = opNotSupported t.DType
//    override t.ExpT() = opNotSupported t.DType
//    override t.LogT() = opNotSupported t.DType
//    override t.Log10T() = opNotSupported t.DType
//    override t.SqrtT() = opNotSupported t.DType
//    override t.SinT() = opNotSupported t.DType
//    override t.CosT() = opNotSupported t.DType
//    override t.TanT() = opNotSupported t.DType
//    override t.SinhT() = opNotSupported t.DType
//    override t.CoshT() = opNotSupported t.DType
//    override t.TanhT() = opNotSupported t.DType
//    override t.AsinT() = opNotSupported t.DType
//    override t.AcosT() = opNotSupported t.DType
//    override t.AtanT() = opNotSupported t.DType

//type RawTensorBoolCPUStatics() = 

//    inherit RawTensorStatics()

//    override _.Zero = upcast  RawTensorBoolCPU([| false |], [||])
//    override _.One = upcast RawTensorBoolCPU([| true |], [||])
//    override _.Zeros(shape:int[]) = upcast RawTensorBoolCPU(Array.zeroCreate (shapeLength shape), shape)
//    override _.Ones(shape:int[]) = upcast RawTensorBoolCPU(Array.create (shapeLength shape) true, shape)
//    override _.Full(shape:int[], value:obj) = upcast (RawTensorCPU.Full (shape, System.Convert.ToBoolean value) |> RawTensorBoolCPU)
//    override _.Random(shape:int[]) = upcast (RawTensorCPU.Random (fun x -> x > 0.5) shape |> RawTensorBoolCPU)
//    override _.RandomNormal(shape:int[]) = upcast (RawTensorCPU.Random (fun x -> x > 0.5) shape |> RawTensorBoolCPU)
//    override _.CreateFromFlatArray(values:Array, shape) = upcast (RawTensorCPU.CreateFromFlatArray (values, shape) |> RawTensorBoolCPU)

