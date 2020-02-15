namespace DiffSharp.Backend

open DiffSharp.Util

type Device =
    | CPU
    | GPU

    member internal x.Code =
        match x with
        | CPU -> 0x000
        | GPU -> 0x001

    member internal x.Name =
        match x with
        | CPU -> "CPU"
        | GPU -> "GPU"

[<RequireQualifiedAccess>]
type Backend =
    | None
    | OpenBLAS
    | Torch

    member internal x.Code = 
        match x with 
        | None -> 0x000
        | OpenBLAS  -> 0x010
        | Torch -> 0x020

    member x.Name = 
        match x with 
        | None -> "None"
        | OpenBLAS -> "OpenBLAS"
        | Torch -> "Torch"

type DType =
    //| Float16
    | Float32
    | Float64
    | Int32

    member internal x.Code =
        match x with
        //| Float16 -> 0x000
        | Float32  -> 0x100
        | Float64 -> 0x200
        | Int32 -> 0x400

    member internal x.Name =
        match x with
        //| Float16 -> "Float16"
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int32 -> "Int32"

type [<AbstractClass>]
     RawTensorStatics() = 
    static let backends = System.Collections.Concurrent.ConcurrentDictionary<int, RawTensorStatics>()

    abstract Zero : RawTensor
    abstract Zeros : shape:int[] -> RawTensor
    abstract One : RawTensor
    abstract Ones : shape:int[] -> RawTensor
    abstract Random : shape:int[] -> RawTensor
    abstract RandomNormal : shape:int[]-> RawTensor
    abstract Create : obj -> RawTensor

    static member Get(?dtype: DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend Backend.None
        let code = dtype.Code + device.Code + backend.Code
        match backends.TryGetValue(code) with 
        | true, v -> v
        | false, _ -> 
            backends.GetOrAdd(code, fun _ -> 
                let name = "DiffSharp.Backend." + backend.Name
                let fullName = System.Reflection.Assembly.GetExecutingAssembly().FullName.Replace("DiffSharp.Core", name)
                let asm = 
                    try System.Reflection.Assembly.Load(fullName)
                    with e ->  failwithf "Couldn't find assembly '%s', error = %s" fullName (e.ToString())
                let typeName = sprintf "DiffSharp.Backend.%s.RawTensor%s%sStatics" backend.Name dtype.Name device.Name
                let theType = asm.GetType(typeName)
                if isNull theType then failwithf "Couldn't find type '%s' in assembly '%s'" typeName fullName
                match System.Activator.CreateInstance(theType) with
                | :? RawTensorStatics as obj -> obj
                | _ -> failwithf "Found the type '%s' in assembly '%s' but it didn't implement RawTensorStatics" typeName fullName
                ) 

and [<AbstractClass>]
    RawTensor(shape:int[], dtype:DType, device:Device, backend:Backend) =
    member t.Shape = shape
    member t.Dim = shape.Length
    member t.Nelement = shapeLength shape
    member t.DType = dtype
    member t.Device = device
    member t.Backend = backend
    override t.ToString() = t.GetString()
    member t.Extend(shape) = t.CreateFromScalar(t.ToValue(), shape)

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

    static member Random(shape, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Random(shape|>Seq.toArray)

    static member RandomNormal(shape, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.RandomNormal(shape|>Seq.toArray)

    static member Create(values: obj, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Create(values)

    abstract CompareTo: RawTensor -> int
    abstract Create : values: obj -> RawTensor
    abstract CreateFromScalar : value: obj * shape: int[] -> RawTensor
    abstract StackTs: seq<RawTensor> -> RawTensor
    abstract UnstackT: unit -> seq<RawTensor>
    abstract Zero : unit -> RawTensor
    abstract Zeros : int[] -> RawTensor
    abstract One : unit -> RawTensor
    abstract Ones : int[] -> RawTensor
    abstract Random : int[] -> RawTensor
    abstract RandomNormal : int[] -> RawTensor
    abstract RandomMultinomial: int -> RawTensor
    abstract GetString : unit -> string
    abstract GetItem: int[] -> RawTensor
    abstract GetSlice: int[,] -> RawTensor
    abstract ToValue: unit -> obj
    abstract ToArray: unit -> System.Array
    abstract Equals: RawTensor -> bool
    abstract Cast : DType -> RawTensor
    abstract ApproximatelyEquals: RawTensor * float -> bool
    abstract LtTT: RawTensor -> RawTensor
    abstract GtTT: RawTensor -> RawTensor
    abstract LeTT: RawTensor -> RawTensor
    abstract GeTT: RawTensor -> RawTensor
    abstract MaxIndexT : unit -> int[]
    abstract MinIndexT : unit -> int[]
    abstract AddTT : RawTensor -> RawTensor
    abstract AddTT0 : RawTensor -> RawTensor
    abstract AddT2T1: RawTensor -> RawTensor
    abstract AddTTSlice: int[] * RawTensor -> RawTensor
    abstract SubTT : RawTensor -> RawTensor
    abstract SubT0T : RawTensor -> RawTensor
    abstract SubTT0 : RawTensor -> RawTensor
    abstract MulTT : RawTensor -> RawTensor
    abstract MulTT0 : RawTensor -> RawTensor
    abstract DivTT : RawTensor -> RawTensor
    abstract DivT0T : RawTensor -> RawTensor
    abstract DivTT0 : RawTensor -> RawTensor
    abstract PowTT : RawTensor -> RawTensor
    abstract PowT0T: RawTensor -> RawTensor
    abstract PowTT0 : RawTensor -> RawTensor
    abstract MatMulT2T2: RawTensor -> RawTensor
    abstract Conv1D: RawTensor * int * int -> RawTensor
    abstract NegT : unit -> RawTensor
    abstract SumT : unit -> RawTensor
    abstract SumT2Dim0 : unit -> RawTensor
    abstract TransposeT2: unit -> RawTensor
    abstract SqueezeT: int -> RawTensor
    abstract UnsqueezeT: int -> RawTensor
    abstract ViewT: int[] -> RawTensor
    abstract SignT: unit -> RawTensor
    abstract FloorT: unit -> RawTensor
    abstract CeilT: unit -> RawTensor
    abstract RoundT: unit -> RawTensor
    abstract AbsT: unit -> RawTensor
    abstract ReluT: unit -> RawTensor
    abstract SigmoidT: unit -> RawTensor
    abstract ExpT: unit -> RawTensor
    abstract LogT: unit -> RawTensor
    abstract Log10T: unit -> RawTensor
    abstract SqrtT: unit -> RawTensor
    abstract SinT: unit -> RawTensor
    abstract CosT: unit -> RawTensor
    abstract TanT: unit -> RawTensor
    abstract SinhT: unit -> RawTensor
    abstract CoshT: unit -> RawTensor
    abstract TanhT: unit -> RawTensor
    abstract AsinT: unit -> RawTensor
    abstract AcosT: unit -> RawTensor
    abstract AtanT: unit -> RawTensor
