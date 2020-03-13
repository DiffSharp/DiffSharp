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
    | Float16
    | Float32
    | Float64

    member internal x.Code =
        match x with
        | Float16 -> 0x000
        | Float32  -> 0x100
        | Float64 -> 0x200

    member internal x.Name =
        match x with
        | Float16 -> "Float16"
        | Float32 -> "Float32"
        | Float64 -> "Float64"

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

    static member Create(obj: obj, ?dtype, ?device, ?backend) =
        let statics = RawTensorStatics.Get(?dtype=dtype, ?device=device, ?backend=backend)
        statics.Create(obj)

    abstract member CompareTo: RawTensor -> int
    abstract member Create : obj -> RawTensor
    abstract member CreateFromScalar : obj * int[] -> RawTensor
    abstract member Clone : unit -> RawTensor
    abstract member Expand: newShape: int[] -> RawTensor
    abstract member StackTs: RawTensor[] -> RawTensor
    abstract member UnstackT: unit -> RawTensor[]
    abstract member Zero : unit -> RawTensor
    abstract member Zeros : int[] -> RawTensor
    abstract member One : unit -> RawTensor
    abstract member Ones : int[] -> RawTensor
    abstract member Random : int[] -> RawTensor
    abstract member RandomNormal : int[] -> RawTensor
    abstract member RandomMultinomial: int -> RawTensor
    abstract member GetString : unit -> string
    abstract member GetItem: int[] -> RawTensor
    abstract member GetSlice: int[,] -> RawTensor
    abstract member ToValue: unit -> obj
    abstract member ToArray: unit -> System.Array
    abstract member Equals: RawTensor -> bool
    abstract member ComputeHash: unit -> int
    abstract member ApproximatelyEquals: RawTensor * float -> bool
    abstract member LtTT: RawTensor -> RawTensor
    abstract member GtTT: RawTensor -> RawTensor
    abstract member LeTT: RawTensor -> RawTensor
    abstract member GeTT: RawTensor -> RawTensor
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
    abstract member Conv1D: RawTensor * int * int -> RawTensor
    abstract member Conv2D: RawTensor * int[] * int[] -> RawTensor
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