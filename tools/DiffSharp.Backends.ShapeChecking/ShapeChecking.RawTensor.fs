namespace rec DiffSharp.Backends.ShapeChecking

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

type ShapeCheckingTensor(shape: Shape, dtype: Dtype, device: Device) =
    inherit RawTensor()

    let sample =
        match dtype with 
        | Float16 -> 0.0f :> scalar
        | BFloat16 -> 0.0f :> scalar
        | Float32 -> 0.0f :> scalar
        | Float64 -> 0.0 :> scalar
        | Int8 -> 0y :> scalar
        | Byte -> 0uy :> scalar
        | Int16 -> 0s :> scalar
        | Int32 -> 0 :> scalar
        | Int64 -> 0L :> scalar
        | Bool -> false :> scalar

    let mutable isMutable = false
    let checkMutable() = if not isMutable then failwith "the tensor can't be mutated" 
    member t.MakeLike(?shape: Shape, ?device: Device, ?dtype) =
        ShapeCheckingTensor(defaultArg shape t.Shape, defaultArg dtype t.Dtype, defaultArg device t.Device) :> RawTensor

    override _.Shape = shape
    override _.Dim = shape.Length
    override _.Nelement = Shape.nelement shape
    override _.Dtype = dtype
    override _.Device = device
    override _.DeviceType = device.DeviceType
    override _.Handle = box 0
    override _.Backend = Backend.ShapeChecking

    override t.GetItem(_indexes) =
        match t.Shape.TryGetSymScope() with
        | None -> 
            printfn "Value not available for symbolic tensor"
        | Some syms -> 
            syms.ReportDiagnostic (1, sprintf """A construct required a value from a symbolic tensor. Consider either 

    - Changing your ShapeCheck to use concrete inputs, rather than symbolic, or
    - Adjust the construct to propagate symbolic information, or
    - Adjust your model to avoid dynamic dependencies on model inputs, or
    - Add a check for symbolic tensor shapes, e.g. 'if tensor.symbolic then <return-dummy-tensor> else <main-code>'

    Call stack: %A""" (System.Diagnostics.StackTrace(fNeedFileInfo=true).ToString()))
        sample

    override t.GetSlice(fullBounds:Int[,]) =
        let shape = Shape.checkCanGetSlice t.Shape fullBounds
        t.MakeLike(shape)

    override t.Clone() = t :> RawTensor

    override _.ComputeHash() = hash shape 
    
    override t.Expand(newShape) = t.MakeLike(newShape)

    override t.GetString(extra) =
        let sb = System.Text.StringBuilder()
        sb.Append "tensor(" |> ignore
        sb.Append (t.Shape.ToString()) |> ignore
        if t.Dtype <> Dtype.Default then
            sb.Append ",dtype=" |> ignore
            sb.Append (t.Dtype.ToString()) |> ignore
        if t.Device <> Device.Default then
            sb.Append ",device=" |> ignore
            sb.Append (t.Device.ToString()) |> ignore
        if t.Backend <> Backend.Default then
            sb.Append ",backend=" |> ignore
            sb.Append (t.Backend.ToString()) |> ignore
        sb.Append extra |> ignore
        sb.Append ")" |> ignore
        sb.ToString()

    override t.ToValues() =
        printfn "-----------------\nToValues not available for symbolic, stack trace:\n%s\n------------------\n"  (System.Diagnostics.StackTrace(fNeedFileInfo=true).ToString())
        match t.Dim with
        | 0 -> box sample
        | _ -> 
            let dims = t.Shape.Dims |> Array.map (fun d -> d.ValueOrOne)
            match dtype with 
            | Float16
            | BFloat16
            | Float32 -> ArrayND.init dims (fun _ -> 0.0f)
            | Float64 -> ArrayND.init dims (fun _ -> 0.0)
            | Int8 -> ArrayND.init dims (fun _ -> 0y)
            | Byte -> ArrayND.init dims (fun _ -> 0uy)
            | Int16 -> ArrayND.init dims (fun _ -> 0s)
            | Int32 -> ArrayND.init dims (fun _ -> 0)
            | Int64 -> ArrayND.init dims (fun _ -> 0L)
            | Bool -> ArrayND.init dims (fun _ -> false)

    override _.StackTs(tensors, dim) =
        let shapes = tensors |> Array.map (fun t -> t.Shape)
        let _, _, _, newShape = Shape.checkCanStack shapes dim
        (tensors.[0] :?> ShapeCheckingTensor).MakeLike(newShape)

    override t.UnstackT(dim) =
        let shape = t.Shape
        let _, _, unstackedShape = Shape.checkCanUnstack shape dim
        let n = shape.[dim].Value // the value must be known to do an unstack
        Array.init n (fun _ -> t.MakeLike(unstackedShape))

    override t.CatTs(tensors, dim) =
        let shapes = tensors |> Array.map (fun t -> t.Shape)
        let _, _, _, _, outShape = Shape.checkCanCat shapes dim
        t.MakeLike(outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        outShapes |> Array.map (fun outShape -> t.MakeLike(outShape))

    override t.TransposeT(dim0, dim1) =
        Shape.checkCanTranspose t.Shape dim0 dim1
        let shape = Array.copy t.Shape.Dims
        shape.[dim0] <- t.Shape.[dim1]
        shape.[dim1] <- t.Shape.[dim0]
        t.ZerosLike(Shape shape)

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        t.MakeLike(Shape [| t.Shape.[1]; t.Shape.[0]|])

    override t.SqueezeT(dim) =
        t.MakeLike(Shape.squeeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        t.MakeLike(outputShape)

    override t.FlipT(dims:int[]) =
        Shape.checkCanFlip t.Dim dims
        t :> RawTensor

    override t.DilateT(dilations:Int[]) =
        Shape.checkCanDilate t.Dim dilations
        t.MakeLike(Shape.dilated t.Shape dilations) 

    override t.UndilateT(dilations:Int[]) =
        t.MakeLike(Shape.undilatedShape t.Shape dilations)

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype
        t.MakeLike(indices.Shape) 

    override t.ViewT(shape:Shape) =
        Shape.checkCanView t.Shape shape
        t.MakeLike(shape)

    override t.Cast(dtype: Dtype) = t.MakeLike(dtype=dtype)

    override t.MoveTo(device: Device) = t.MakeLike(shape, device=device)

    override t1.Equals(_t2:RawTensor) : bool =  printfn "Equals"; false
    override t1.AllClose(_t2:RawTensor, _relativeTolerance, _absoluteTolerance) = printfn "AllClose"; false
    override t.ClampT(_low, _high) = t.MakeLike()
    override t.SoftplusT() = t.MakeLike()
    override t1.LtTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.GtTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.LeTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.GeTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.EqTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.NeqTT(_t2) = t1.MakeLike(dtype=Bool)
    override t.MaxIndexT() = Array.zeroCreate t.Dim
    override t.MinIndexT() = Array.zeroCreate t.Dim
    override t1.AddTT(_t2, _alpha) = t1.MakeLike()
    override t1.AddTT0(_t2, _alpha) = t1.MakeLike()
    override t1.AddTTSlice(_location:Int[], _t2) = t1.MakeLike()
    override t1.SubTT(_t2) = t1.MakeLike()
    override t1.SubFromT0T(_t2) = t1.MakeLike()
    override t1.SubTT0(_t2) = t1.MakeLike()
    override t1.MulTT(_t2) = t1.MakeLike()
    override t1.MulTT0(_t2) = t1.MakeLike()
    override t1.DivTT(_t2) = t1.MakeLike()
    override t1.DivFromT0T(_t2) = t1.MakeLike()
    override t1.DivTT0(_t2) = t1.MakeLike()
    override t1.PowTT(_t2) = t1.MakeLike()
    override t1.PowFromT0T(_t2) = t1.MakeLike()
    override t1.PowTT0(_t2) = t1.MakeLike()

    override _.ClampInPlace(_low, _high) = checkMutable()
    override _.LtInPlace(_t2) = checkMutable()
    override _.GtInPlace(_t2) = checkMutable()
    override _.LeInPlace(_t2) = checkMutable()
    override _.GeInPlace(_t2) = checkMutable()
    override _.EqInPlace(_t2) = checkMutable()
    override _.NeqInPlace(_t2) = checkMutable()
    override _.AddInPlace(_t2, _alpha) = checkMutable()
    override _.AddScalarInPlace(_t2) = checkMutable()
    override _.AddSliceInPlace(_location, _t2) = checkMutable()
    override _.SubInPlace(_t2) = checkMutable()
    override _.SubScalarInPlace(_t2) = checkMutable()
    override _.MulInPlace(_t2) = checkMutable()
    override _.MulScalarInPlace(_t2) = checkMutable()
    override _.DivInPlace(_t2) = checkMutable()
    override _.DivScalarInPlace(_t2) = checkMutable()
    override _.PowInPlace(_t2) = checkMutable()
    override _.PowScalarInPlace(_t2) = checkMutable()
    override _.MatMulInPlace(_t2) = checkMutable()
    override _.NegInPlace() = checkMutable()
    override _.SignInPlace() = checkMutable()
    override _.FloorInPlace() = checkMutable()
    override _.CeilInPlace() = checkMutable()
    override _.RoundInPlace() = checkMutable()
    override _.AbsInPlace() = checkMutable()
    override _.ReluInPlace() = checkMutable()
    override _.SoftplusInPlace() = checkMutable()
    override _.SigmoidInPlace() = checkMutable()
    override _.ExpInPlace() = checkMutable()
    override _.LogInPlace() = checkMutable()
    override _.Log10InPlace() = checkMutable()
    override _.SqrtInPlace() = checkMutable()
    override _.SinInPlace() = checkMutable()
    override _.CosInPlace() = checkMutable()
    override _.TanInPlace() = checkMutable()
    override _.SinhInPlace() = checkMutable()
    override _.CoshInPlace() = checkMutable()
    override _.TanhInPlace() = checkMutable()
    override _.AsinInPlace() = checkMutable()
    override _.AcosInPlace() = checkMutable()
    override _.AtanInPlace() = checkMutable()
    override _.OnesInPlace() = checkMutable()
    override _.RandomInPlace() = checkMutable()
    override _.RandomNormalInPlace() = checkMutable()
    override _.RandomIntInPlace(_low, _high) = checkMutable()
    override _.ZerosInPlace() = checkMutable()

    override t1.MatMulTT(t2) = 
        let (t1BatchPart, t1MatrixPart), (t2BatchPart, t2MatrixPart) = Shape.checkCanMatmul t1.Shape t2.Shape
        if not (Shape t1BatchPart =~= Shape t2BatchPart) then failwithf "Cannot matrix multiply raw tensors with shapes %A, %A - mismatch batching" t1.Shape t2.Shape
        let t1rows = t1MatrixPart.[0]
        let t2cols = t2MatrixPart.[1]
        t1.MakeLike(Shape [| t1rows; t2cols |])

    override t1.MaxPool1D(kernelSize, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanMaxpool1d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxPool2D(kernelSize, stride, padding) =
        let _, _, _, _, _, outputShape = Shape.checkCanMaxpool2d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxPool3D(kernelSize, stride, padding) =
        let _, _, _, _, _, outputShape = Shape.checkCanMaxpool3d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxUnpool1D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool1d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.MaxUnpool2D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool2d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.MaxUnpool3D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool3d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.Conv1D(t2, stride, padding) = 
        let _, _, _, _, _, outputShape = Shape.checkCanConv1d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding 1I
        t1.MakeLike(outputShape)

    override t1.Conv2D(t2, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanConv2d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1I;1I|]
        t1.MakeLike(outputShape) 

    override t1.Conv3D(t2, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanConv3d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1I;1I;1I|]  
        t1.MakeLike(outputShape) 

    override t.NegT() = t :> _

    override t.SumT(resultType) =
        match resultType with 
        | None -> t.MakeLike(Shape.scalar)
        | Some dtype -> t.MakeLike(Shape.scalar, dtype=dtype)
    override t.SumT2Dim0() = t.MakeLike(Shape [|t.Shape.[1]|])
    override t.SignT() = t.MakeLike()
    override t.FloorT() = t.MakeLike()
    override t.CeilT() = t.MakeLike()
    override t.RoundT() = t.MakeLike()
    override t.AbsT() = t.MakeLike()
    override t.ReluT() = t.MakeLike()
    override t.SigmoidT() = t.MakeLike()
    override t.ExpT() = t.MakeLike()
    override t.LogT() = t.MakeLike()
    override t.Log10T() = t.MakeLike()
    override t.SqrtT() = t.MakeLike()
    override t.SinT() = t.MakeLike()
    override t.CosT() = t.MakeLike()
    override t.TanT() = t.MakeLike()
    override t.SinhT() = t.MakeLike()
    override t.CoshT() = t.MakeLike()
    override t.TanhT() = t.MakeLike()
    override t.AsinT() = t.MakeLike()
    override t.AcosT() = t.MakeLike()
    override t.AtanT() = t.MakeLike()
    override t.SetMutable() = isMutable <- true
    override t.IsMutable = isMutable

type ShapeCheckingBackendTensorStatics() = 

    inherit BackendTensorStatics()

    override _.GetDevices(_deviceType) = [ Device.CPU; Device.GPU ]
    override _.IsDeviceTypeSupported (_deviceType) = true
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(dtype, device) = ShapeCheckingTensor(Shape.scalar, dtype, device) :> _
    override _.One(dtype, device) = ShapeCheckingTensor(Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:Shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.Empty(shape:Shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.Ones(shape:Shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.Full(shape:Shape, _value:scalar, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.Random(shape:Shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.RandomNormal(shape:Shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.RandomInt(shape:Shape, _low:int, _high:int, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
    override _.CreateFromFlatArray(_values:Array, shape, dtype, device) = ShapeCheckingTensor(shape, dtype, device) :> _
