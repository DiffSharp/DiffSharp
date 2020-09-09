namespace rec DiffSharp.Backends

open System
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type BackendStatics() = 
    // cache for most recently accessed backend
    static let mutable last = None
    static let backends = System.Collections.Concurrent.ConcurrentDictionary<int, BackendStatics>()

    abstract Seed: seed:int -> unit
    abstract Zero: device: Device -> RawTensor
    abstract Zeros: shape:int[] * device: Device -> RawTensor
    abstract One: device: Device -> RawTensor
    abstract Ones: shape:int[] * device: Device -> RawTensor
    abstract Full: shape:int[] * obj * device: Device -> RawTensor
    abstract Random: shape:int[] * device: Device -> RawTensor
    abstract RandomNormal: shape:int[] * device: Device -> RawTensor
    abstract RandomInt: shape:int[] * low:int * high:int * device: Device -> RawTensor
    abstract GetDevices: ?deviceType: DeviceType -> Device list
    abstract IsDeviceTypeSupported: deviceType: DeviceType -> bool
    
    static member Seed(?seed:int) =
        let seed = defaultArg seed (int DateTime.Now.Ticks)
        Random.Seed(seed) // Do not remove. util.Random seed would be set by the Reference backend if it's currently loaded. However we still need to keep this here to ensure util.Random seed is set (it may be used in code other than the Reference backend).
        for KeyValue(_, backend) in backends do
            backend.Seed(seed)

    /// Create a tensor of appropriate dtype from a scalar or array of appropriate values.
    /// A backend type is delivered consistent with in-memory data - a type for dtype Int32 gets int32 data etc.
    abstract CreateFromFlatArray: data: System.Array * shape: int[] * device: Device -> RawTensor

    static member Get(?dtype: Dtype, ?backend: Backend) =
        // Note we re-examing the default backends etc. each time we create a root tensor.
        let dtype = defaultArg dtype Dtype.Default
        let backend = defaultArg backend Backend.Default
        let code = dtype.Code + backend.Code
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
                    let typeName = sprintf "DiffSharp.Backends.%s.%s%sStatics" backend.Name backend.Name dtype.Name
                    let theType = asm.GetType(typeName)
                    if isNull theType then failwithf "Couldn't find type '%s' in assembly '%s'" typeName fullName
                    let obj = 
                        match System.Activator.CreateInstance(theType) with
                        | :? BackendStatics as obj -> obj
                        | _ -> failwithf "Found the type '%s' in assembly '%s' but it didn't implement BackendStatics" typeName fullName
                    obj
                    ) 
            last <- Some (code, res)
            res

[<AbstractClass>]
type RawTensor() =

    //let dummy1 = Array.copy shape
    //let dummy2 = Array.copy shape
    //let dummy3 = Array.copy shape
    //let dummy4 = Array.copy shape

    //member t.Dummy = (dummy1, dummy2, dummy3, dummy4)
    abstract member Shape : int[]
    abstract member Dim : int
    abstract member Nelement : int
    abstract member Dtype : Dtype
    abstract member Device : Device
    abstract member DeviceType : DeviceType
    abstract member Backend : Backend
    override t.ToString() = t.GetString()
    
    static member Zero(?dtype, ?device, ?backend) = 
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.Zero(device)

    static member Zeros(shape, ?dtype, ?device, ?backend) = 
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.Zeros(shape, device)

    static member One(?dtype, ?device, ?backend) = 
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.One(device)

    static member Ones(shape, ?dtype, ?device, ?backend) =
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.Ones(shape, device)

    static member Full(shape, value, ?dtype, ?device, ?backend) =
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.Full(shape, value, device)

    static member Random(shape, ?dtype, ?device, ?backend) =
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.Random(shape, device)

    static member RandomNormal(shape, ?dtype, ?device, ?backend) =
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.RandomNormal(shape, device)

    static member RandomInt(shape, low, high, ?dtype, ?device, ?backend) =
        let statics = BackendStatics.Get(?dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default
        statics.RandomInt(shape|>Seq.toArray, low, high, device)

    static member Create(values: obj, ?dtype, ?device, ?backend) =
        // We deliver consistent in-memory data to the backend - a dtype Int32 gets int32 etc.
        let data, shape, dtype =
            match dtype with 
            | Some Dtype.Int64 ->
                let a,s = DataConverter.dataOfValuesForInt64 values
                (a :> Array), s, Dtype.Int64
            | Some Dtype.Int32 ->
                let a,s = DataConverter.dataOfValuesForInt32 values
                (a :> Array), s, Dtype.Int32
            | Some Dtype.Int16 ->
                let a,s = DataConverter.dataOfValuesForInt16 values
                (a :> Array), s, Dtype.Int16
            | Some Dtype.Int8 ->
                let a,s = DataConverter.dataOfValuesForInt8 values
                (a :> Array), s, Dtype.Int8
            | Some Dtype.Byte ->
                let a,s = DataConverter.dataOfValuesForByte values
                (a :> Array), s, Dtype.Byte
            | Some Dtype.Bool ->
                let a,s = DataConverter.dataOfValuesForBool values
                (a :> Array), s, Dtype.Bool
            | Some Dtype.Float64 ->
                let a,s = DataConverter.dataOfValuesForFloat64 values
                (a :> Array), s, Dtype.Float64
            | Some Dtype.Float32 ->
                let a,s = DataConverter.dataOfValuesForFloat32 values
                (a :> Array), s, Dtype.Float32
            | None ->
                // Prefer Bool tensor if all bool
                match values |> DataConverter.tryFlatArrayAndShape<bool> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Bool)
                | _ ->
                // Otherwise prefer float32
                let a,s = DataConverter.dataOfValuesForFloat32 values 
                (a :> Array), s, Dtype.Float32

        let statics = BackendStatics.Get(dtype=dtype, ?backend=backend)
        let device = defaultArg device Device.Default

        statics.CreateFromFlatArray(data, shape, device)

    member t.CreateLike(values: obj, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Create(values, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.ZeroLike(?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Zero(dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.ZerosLike(shape: int[], ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Zeros(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.OneLike(?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.One(dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.OnesLike(shape: int[], ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Ones(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.FullLike(shape: int[], value: obj, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Full(shape, value, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.RandomLike(shape: int[], ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Random(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.RandomNormalLike(shape: int[], ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.RandomNormal(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    member t.RandomIntLike(shape: int[], low:int, high:int, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.RandomInt(shape=shape, low=low, high=high, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    abstract member Clone : unit -> RawTensor
    abstract member Expand: newShape: int[] -> RawTensor
    abstract member StackTs: RawTensor[] * dim:int -> RawTensor
    abstract member UnstackT: dim:int -> RawTensor[]
    abstract member CatTs: RawTensor[] * dim: int -> RawTensor
    abstract member SplitT: int[] * dim: int -> RawTensor[]
    abstract member GetString: unit -> string
    
    /// The indexes are an Nx3 array.   The first row is the start bounds, the second row is
    /// the end bounds, the third is 1/0 indicating dimension removal.
    abstract member GetSlice: int[,] -> RawTensor

    abstract member ToValues: unit -> obj
    abstract member Equals: RawTensor -> bool
    abstract member Cast : Dtype -> RawTensor
    abstract member MoveTo : Device -> RawTensor
    abstract member ComputeHash: unit -> int
    abstract member AllClose: RawTensor * float * float -> bool
    abstract member ClampT: RawTensor * RawTensor -> RawTensor
    abstract member GatherT: int * RawTensor -> RawTensor
    abstract member LtTT: RawTensor -> RawTensor
    abstract member GtTT: RawTensor -> RawTensor
    abstract member LeTT: RawTensor -> RawTensor
    abstract member GeTT: RawTensor -> RawTensor
    abstract member EqTT: RawTensor -> RawTensor
    abstract member NeqTT: RawTensor -> RawTensor
    abstract member IsInfT : unit -> RawTensor
    abstract member IsNaNT : unit -> RawTensor
    abstract member GetItem : [<System.ParamArray>] indexes: int[] -> obj 
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
    abstract member MatMulTT: RawTensor -> RawTensor
    abstract member MaxPool1D: int * int * int -> RawTensor * RawTensor
    abstract member MaxPool2D: int[] * int[] * int[] -> RawTensor * RawTensor
    abstract member MaxPool3D: int[] * int[] * int[] -> RawTensor * RawTensor
    abstract member MaxUnpool1D: RawTensor * int[] -> RawTensor
    abstract member MaxUnpool2D: RawTensor * int[] -> RawTensor
    abstract member MaxUnpool3D: RawTensor * int[] -> RawTensor
    abstract member Conv1D: RawTensor * int * int -> RawTensor
    abstract member Conv2D: RawTensor * int[] * int[] -> RawTensor
    abstract member Conv3D: RawTensor * int[] * int[] -> RawTensor
    abstract member NegT : unit -> RawTensor
    abstract member SumT : ?resultType: Dtype -> RawTensor
    abstract member SumT2Dim0 : unit -> RawTensor
    abstract member TransposeT: int * int -> RawTensor
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

    default t.IsInfT() =
        match t.Dtype with 
        | Dtype.IntegralOrBool -> t.FullLike(t.Shape, false, dtype=Dtype.Bool)
        | _ -> t.AbsT().EqTT(t.FullLike(t.Shape,System.Single.PositiveInfinity))

    default t.IsNaNT() =
        match t.Dtype with 
        | Dtype.IntegralOrBool -> t.FullLike(t.Shape, false, dtype=Dtype.Bool)
        | _ -> t.NeqTT(t)

    default t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        let printVal (x:obj) = 
           match x with 
           | :? single as v -> sprintf "%f" v
           | :? double as v -> sprintf "%f" v
           | :? byte as v -> sprintf "%d" v
           | :? int8 as v -> sprintf "%d" v
           | :? int16 as v -> sprintf "%d" v
           | :? int32 as v -> sprintf "%d" v
           | :? int64 as v -> sprintf "%d" v
           | :? bool as v -> if v then "true" else "false"
           | _ -> sprintf "%A" x

        match t.Dim with
        | 0 -> printVal (t.ToScalar())
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(printVal (t.GetItem(globalCoords))) |> ignore
                        prefix <- ", "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    let prefix2 = sprintf ", %s%s" (String.replicate (max 1 (shape.Length-1)) "\n") (String.replicate (externalCoords.Length+1) " ")
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- prefix2
                    sb.Append("]") |> ignore
            print t.Shape [||]
            sb.ToString()

    override x.Equals(yobj: obj) = 
        match yobj with
        | :? RawTensor as y -> x.Equals(y)
        | _ -> false

    override x.GetHashCode() = x.ComputeHash()

    interface System.IComparable with 
        member x.CompareTo(yobj) =
            match yobj with
            | :? RawTensor as y -> Unchecked.compare (x.ToScalar()) (y.ToScalar())
            | _ -> failwithf "cannot compare RawTensor with object of type %A" (yobj.GetType())

    member t.ToScalar() =
        match t.Dim with
        | 0 -> t.ToValues()
        | _ -> failwithf "Cannot convert %Ad Tensor to scalar" t.Dim

    member t.ToArray() =
        match t.Dim with
        | 0 -> failwithf "Cannot convert scalar Tensor to array"
        | _ ->
            match t.ToValues() with 
            | :? System.Array as a -> a
            | _ -> failwithf "ToValue() should return an array but returned type %A" (t.GetType())

