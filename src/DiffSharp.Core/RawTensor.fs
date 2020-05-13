namespace DiffSharp.Backends

open System
open DiffSharp
open DiffSharp.Util

type [<AbstractClass>]
     RawTensorStatics() = 
    // cache for most recently accessed backend
    static let mutable last = None
    static let backends = System.Collections.Concurrent.ConcurrentDictionary<int, RawTensorStatics>()

    abstract Zero: RawTensor
    abstract Zeros: shape:int[] -> RawTensor
    abstract One: RawTensor
    abstract Ones: shape:int[] -> RawTensor
    abstract Full: shape:int[] * obj -> RawTensor
    abstract Random: shape:int[] -> RawTensor
    abstract RandomNormal: shape:int[] -> RawTensor
    
    /// Create a tensor of appropriate dtype from a scalar or array of appropriate values.
    /// A backend type is delivered consistent in-memory data - a type for dtype Int32 gets int32 data etc.
    abstract CreateFromFlatArray: data: System.Array * shape: int[] -> RawTensor

    static member Get(?dtype: DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend Backend.None
        let code = dtype.Code + device.Code + backend.Code
        match last with 
        | Some (code2, v) when code = code2 -> v
        | _ ->
        match backends.TryGetValue(code) with 
        | true, v -> v
        | false, _ -> 
            let res =
                backends.GetOrAdd(code, fun _ -> 
                    let name = "DiffSharp.Backends." + backend.Name
                    let fullName = System.Reflection.Assembly.GetExecutingAssembly().FullName.Replace("DiffSharp.Core", name)
                    let asm = 
                        try System.Reflection.Assembly.Load(fullName)
                        with e ->  failwithf "Couldn't find assembly '%s', error = %s" fullName (e.ToString())
                    let typeName = sprintf "DiffSharp.Backends.%s.RawTensor%s%sStatics" backend.Name dtype.Name device.Name
                    let theType = asm.GetType(typeName)
                    if isNull theType then failwithf "Couldn't find type '%s' in assembly '%s'" typeName fullName
                    let obj = 
                        match System.Activator.CreateInstance(theType) with
                        | :? RawTensorStatics as obj -> obj
                        | _ -> failwithf "Found the type '%s' in assembly '%s' but it didn't implement RawTensorStatics" typeName fullName
                    obj
                    ) 
            last <- Some (code, res)
            res

and [<AbstractClass>]
    RawTensor(shape:int[], dtype:DType, device:Device, backend:Backend) =

    member t.Shape = shape
    member t.Dim = shape.Length
    member t.Nelement = shapeLength shape
    member t.DType = dtype
    member t.Device = device
    member t.Backend = backend
    override t.ToString() = t.GetString()
    
    static member Zero(?dtype, ?device, ?backend) = 
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Zero

    static member Zeros(shape, ?dtype, ?device, ?backend) = 
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Zeros(shape|>Seq.toArray)

    static member One(?dtype, ?device, ?backend) = 
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.One

    static member Ones(shape, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Ones(shape|>Seq.toArray)

    static member Full(shape, value, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Full(shape|>Seq.toArray, value)

    static member Random(shape, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Random(shape|>Seq.toArray)

    static member RandomNormal(shape, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.RandomNormal(shape|>Seq.toArray)

    static member Create(values: obj, ?dtype, ?device, ?backend) =
        // We deliver consistent in-memory data to the backend - a dtype Int32 gets int32 etc.
        let data, shape, dtype =
            match dtype with 
            | Some DType.Int64 ->
                let a,s = dataOfValuesForInt64 values
                (a :> Array), s, DType.Int64
            | Some DType.Int32 ->
                let a,s = dataOfValuesForInt32 values
                (a :> Array), s, DType.Int32
            | Some DType.Int16 ->
                let a,s = dataOfValuesForInt16 values
                (a :> Array), s, DType.Int16
            | Some DType.Int8 ->
                let a,s = dataOfValuesForInt8 values
                (a :> Array), s, DType.Int8
            | Some DType.Bool ->
                let a,s = dataOfValuesForBool values
                (a :> Array), s, DType.Bool
            | Some DType.Float64 ->
                let a,s = dataOfValuesForFloat64 values
                (a :> Array), s, DType.Float64
            | Some DType.Float32 ->
                let a,s = dataOfValuesForFloat32 values
                (a :> Array), s, DType.Float32
            | None ->
                // Prefer Bool tensor if all bool
                match values |> tryFlatArrayAndShape<bool> with
                | Some (values, shape) -> ((values :> Array), shape, DType.Bool)
                | _ ->
                // Otherwise prefer float32
                let a,s = dataOfValuesForFloat32 values 
                (a :> Array), s, DType.Float32

        let statics = RawTensorStatics.Get(dtype=dtype, ?device=device, ?backend=backend)

        statics.CreateFromFlatArray(data, shape)

    member t.CreateLike(values: obj, ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Create(values, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.ZeroLike(?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Zero(dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.ZerosLike(shape: int[], ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Zeros(shape=shape, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.OneLike(?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.One(dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.OnesLike(shape: int[], ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Ones(shape=shape, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.FullLike(shape: int[], value: obj, ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Full(shape, value, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.RandomLike(shape: int[], ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.Random(shape=shape, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.RandomNormalLike(shape: int[], ?dtype: DType, ?device: Device, ?backend: Backend) =
        RawTensor.RandomNormal(shape=shape, dtype=defaultArg dtype t.DType, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    abstract member CompareTo: RawTensor -> int
    abstract member Clone : unit -> RawTensor
    abstract member Expand: newShape: int[] -> RawTensor
    abstract member StackTs: RawTensor[] * dim:int -> RawTensor
    abstract member UnstackT: dim:int -> RawTensor[]
    abstract member CatTs: RawTensor[] * dim: int -> RawTensor
    abstract member SplitT: int[] * dim: int -> RawTensor[]
    abstract member GetString: unit -> string
    abstract member GetItem: int[] -> RawTensor
    abstract member GetSlice: int[,] -> RawTensor
    abstract member ToValues: unit -> obj
    abstract member Equals: RawTensor -> bool
    abstract member Cast : DType -> RawTensor
    abstract member ComputeHash: unit -> int
    abstract member RandomMultinomial: numSamples: int -> RawTensor
    abstract member AllClose: RawTensor * float * float -> bool
    abstract member LtTT: RawTensor -> RawTensor
    abstract member GtTT: RawTensor -> RawTensor
    abstract member LeTT: RawTensor -> RawTensor
    abstract member GeTT: RawTensor -> RawTensor
    abstract member IsInfT : unit -> RawTensor
    abstract member IsNaNT : unit -> RawTensor
    abstract member MaxIndexT : unit -> int[]
    abstract member MinIndexT : unit -> int[]
    abstract member AddTT : RawTensor -> RawTensor
    abstract member AddTT0 : RawTensor -> RawTensor
    abstract member AddT2T1: RawTensor -> RawTensor
    abstract member AddTTSlice: int[] * RawTensor -> RawTensor
    abstract member SubTT : RawTensor -> RawTensor
    abstract member SubT0T : RawTensor -> RawTensor
    abstract member SubTT0 : RawTensor -> RawTensor
    abstract member MulTT : RawTensor -> RawTensor
    abstract member MulTT0 : RawTensor -> RawTensor
    abstract member DivTT : RawTensor -> RawTensor
    abstract member DivT0T : RawTensor -> RawTensor
    abstract member DivTT0 : RawTensor -> RawTensor
    abstract member PowTT : RawTensor -> RawTensor
    abstract member PowT0T: RawTensor -> RawTensor
    abstract member PowTT0 : RawTensor -> RawTensor
    abstract member MatMulT2T2: RawTensor -> RawTensor
    abstract member MaxPool1D: int * int * int -> RawTensor * RawTensor
    abstract member MaxUnpool1D: RawTensor * int -> RawTensor
    abstract member Conv1D: RawTensor * int * int -> RawTensor
    abstract member Conv2D: RawTensor * int[] * int[] -> RawTensor
    abstract member Conv3D: RawTensor * int[] * int[] -> RawTensor
    abstract member NegT : unit -> RawTensor
    abstract member SumT : unit -> RawTensor
    abstract member SumT2Dim0 : unit -> RawTensor
    abstract member TransposeT2: unit -> RawTensor
    abstract member SqueezeT: int -> RawTensor
    abstract member UnsqueezeT: int -> RawTensor
    abstract member FlipT: int[] -> RawTensor
    abstract member DilateT: int[] -> RawTensor
    abstract member UndilateT: int[] -> RawTensor
    abstract member ViewT: int[] -> RawTensor
    abstract member SignT: unit -> RawTensor
    abstract member FloorT: unit -> RawTensor
    abstract member CeilT: unit -> RawTensor
    abstract member RoundT: unit -> RawTensor
    abstract member AbsT: unit -> RawTensor
    abstract member ReluT: unit -> RawTensor
    abstract member SoftplusT: unit -> RawTensor
    abstract member SigmoidT: unit -> RawTensor
    abstract member ExpT: unit -> RawTensor
    abstract member LogT: unit -> RawTensor
    abstract member Log10T: unit -> RawTensor
    abstract member SqrtT: unit -> RawTensor
    abstract member SinT: unit -> RawTensor
    abstract member CosT: unit -> RawTensor
    abstract member TanT: unit -> RawTensor
    abstract member SinhT: unit -> RawTensor
    abstract member CoshT: unit -> RawTensor
    abstract member TanhT: unit -> RawTensor
    abstract member AsinT: unit -> RawTensor
    abstract member AcosT: unit -> RawTensor
    abstract member AtanT: unit -> RawTensor

    override x.Equals(yobj: obj) = 
        match yobj with
        | :? RawTensor as y -> x.Equals(y)
        | _ -> false

    override x.GetHashCode() = x.ComputeHash()

    interface System.IComparable with 
        member x.CompareTo(yobj) =
            match yobj with
            | :? RawTensor as y -> x.CompareTo(y)
            | _ -> failwithf "cannot compare RawTensor with object of type %A" (yobj.GetType())

    member t.ToScalar() =
        match t.Dim with
        | 0 -> t.ToValues()
        | _ -> failwithf "Cannot convert %Ad Tensor to scalar" t.Dim

    member t.ToArray() =
        match t.Dim with
        | 0 -> failwithf "Cannot convert scalar Tensor to array"
        | _ ->
            match t.ToValues()with 
            | :? System.Array as a -> a
            | _ -> failwithf "ToValue() should return an array but returned type %A" (t.GetType())

