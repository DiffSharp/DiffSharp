namespace rec DiffSharp

type Device =
    | CPU
    | GPU
    | Other of name:string * code:int

    member internal x.Code =
        match x with
        | CPU -> 0x0000
        | GPU -> 0x0001
        | Other (_name, code) -> (code + 2)

    member internal x.Name =
        match x with
        | CPU -> "CPU"
        | GPU -> "GPU"
        | Other (name, _code) -> name

    static member Default = Device.CPU

module Device = 
    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Device>()
    let Register name = codes.GetOrAdd(name, (fun _ -> incr count; Device.Other(name, count.Value)))

[<RequireQualifiedAccess>]
type Backend =
    | Reference
    | Torch
    | OpenBLAS
    | Other of name: string * code: int

    static member Default = Backend.Reference

    member internal x.Code = 
        match x with 
        | Reference -> 0x000
        | OpenBLAS -> 0x0100
        | Torch -> 0x0200
        | Other (_name, code) -> (code + 3) <<< 8

    member x.Name = 
        match x with 
        | Reference -> "Reference"
        | OpenBLAS -> "OpenBLAS"
        | Torch -> "Torch"
        | Other (name, _) -> name

module Backend = 
    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Backend>()
    let Register name = codes.GetOrAdd(name, (fun _ -> incr count; Backend.Other(name, count.Value)))

type DType =
    | Float32
    | Float64
    | Int8
    | Int16
    | Int32
    | Int64
    | Bool
    | Other of name:string * code:int * inOutType: System.Type

    member internal x.Code =
        match x with
        | Float32 -> 0x10000
        | Float64 -> 0x20000
        | Int8 -> 0x30000
        | Int16 -> 0x40000
        | Int32 -> 0x50000
        | Int64 -> 0x60000
        | Bool -> 0x70000
        | Other (_name, code, _) -> (code + 8) <<< 16

    member internal x.Name =
        match x with
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int8 -> "Int8"
        | Int16 -> "Int16"
        | Int32 -> "Int32"
        | Int64 -> "Int64"
        | Bool -> "Bool"
        | Other (name, _, _) -> name

    static member Default = DType.Float32

    member x.AsType () =
        match x with
        | Float32 -> typeof<single>
        | Float64 -> typeof<double>
        | Int8 -> typeof<int8>
        | Int16 -> typeof<int16>
        | Int32 -> typeof<int32>
        | Int64 -> typeof<int64>
        | Bool -> typeof<bool>
        | Other (_name, _, typ) -> typ

    /// Gets the natural result of the Sum(), SumToSize() and Sum(dim) operation on this dtype
    member t.SummationType =
        match t with
        | Bool | Int8 | Int16 | Int32 | Int64 -> DType.Int64
        | dt -> dt

module DType =

    let (|FloatingPoint|_|) x =
        match x with
        | Float32 | Float64 -> Some()
        | _ -> None

    let (|Integral|_|) x =
        match x with
        | Int8 | Int16 | Int32 | Int64 -> Some()
        | _ -> None

    let (|IntegralOrBool|_|) x =
        match x with
        | Integral | Bool -> Some()
        | _ -> None

    /// Find the DType into which dtype1 and dtype2 can be widened
    let widen (dtype1: DType) (dtype2: DType) =
        if dtype1 = dtype2 then dtype1
        else
            match dtype1, dtype2 with 
            | Other _,_ | _, Other _ ->  failwith "cannot widen user-defined tensor types, must cast explicitly"
            | Float64, _ | _, Float64 -> Float64
            | Float32, _ | _, Float32 -> Float32
            | Int64, _ | _, Int64 -> Int64
            | Int32, _ | _, Int32 -> Int32
            | Int16, _ | _, Int16 -> Int16
            | Int8, _ | _, Int8 -> Int8
            | _ -> Bool

    /// Convert System.Type to DType
    let ofType (ty: System.Type) =
        if ty.Equals(typeof<int32>) then DType.Int32
        elif ty.Equals(typeof<double>) then DType.Float64
        elif ty.Equals(typeof<single>) then DType.Float32
        elif ty.Equals(typeof<int64>) then DType.Int64
        elif ty.Equals(typeof<int16>) then DType.Int16
        elif ty.Equals(typeof<int8>) then DType.Int8
        elif ty.Equals(typeof<bool>) then DType.Bool
        else failwithf "unknown type '%A' used as tensor type" ty

    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,DType>()

    let Register name inOutType = codes.GetOrAdd(name, (fun _ -> incr count; DType.Other(name, count.Value, inOutType)))

[<AutoOpen>]
module DTypeGlobalOps =
    let opNotSupported msg (t: DType) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A" msg t)

    let opNotSupported2 msg (t1: DType) (t2: DType) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type (%A, %A)" msg t1 t2)

