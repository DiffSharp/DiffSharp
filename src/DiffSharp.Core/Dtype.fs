namespace DiffSharp

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

module Device = 
    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Device>()
    let Register name = codes.GetOrAdd(name, (fun _ -> incr count; Device.Other(name, count.Value)))
    let mutable Default = Device.CPU

[<RequireQualifiedAccess>]
type Backend =
    | Reference
    | Torch
    | OpenBLAS
    | Other of name: string * code: int

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
    let mutable Default = Backend.Reference

type Dtype =
    //| Float16
    | Float32
    | Float64
    | Int8
    | Byte
    | Int16
    | Int32
    | Int64
    | Bool
    | Other of name:string * code:int * inOutType: System.Type

    member internal x.Code =
        match x with
        //| Float16 -> 0x10000
        | Float32 -> 0x20000
        | Float64 -> 0x30000
        | Int8 -> 0x40000
        | Byte -> 0x50000
        | Int16 -> 0x60000
        | Int32 -> 0x70000
        | Int64 -> 0x80000
        | Bool -> 0x90000
        | Other (_name, code, _) -> (code + 9) <<< 16

    member internal x.Name =
        match x with
        //| Float16 -> "Float16"
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int8 -> "Int8"
        | Byte -> "Byte"
        | Int16 -> "Int16"
        | Int32 -> "Int32"
        | Int64 -> "Int64"
        | Bool -> "Bool"
        | Other (name, _, _) -> name

    member x.AsType () =
        match x with
        //| Float16 -> typeof<single>
        | Float32 -> typeof<single>
        | Float64 -> typeof<double>
        | Int8 -> typeof<int8>
        | Byte -> typeof<byte>
        | Int16 -> typeof<int16>
        | Int32 -> typeof<int32>
        | Int64 -> typeof<int64>
        | Bool -> typeof<bool>
        | Other (_name, _, typ) -> typ

    /// Gets the natural result of the Sum(), SumToSize() and Sum(dim) operation on this dtype
    member t.SummationType =
        match t with
        | Bool | Byte | Int8 | Int16 | Int32 | Int64 -> Dtype.Int64
        | dt -> dt

module Dtype =

    let (|FloatingPoint|_|) x =
        match x with
        | Float32 | Float64 -> Some()
        | _ -> None

    let (|Integral|_|) x =
        match x with
        | Byte | Int8 | Int16 | Int32 | Int64 -> Some()
        | _ -> None

    let (|IntegralOrBool|_|) x =
        match x with
        | Integral | Bool -> Some()
        | _ -> None

    /// Find the Dtype into which dtype1 and dtype2 can be widened
    let widen (dtype1: Dtype) (dtype2: Dtype) =
        if dtype1 = dtype2 then Some dtype1
        else
            match dtype1, dtype2 with 
            | Other _,_ | _, Other _ ->  None //failwith "cannot widen user-defined tensor types, must cast explicitly"
            | Float64, _ | _, Float64 -> Some Float64
            | Float32, _ | _, Float32 -> Some Float32
            | Int64, _ | _, Int64 -> Some Int64
            | Int32, _ | _, Int32 -> Some Int32
            | Int16, _ | _, Int16 -> Some Int16
            | Int8, Bool | Bool, Int8 -> Some Int8
            | Byte, Bool | Bool, Byte -> Some Byte
            | Int8, Int8 -> Some Int8
            | Byte, Byte -> Some Byte
            | Bool, Bool -> Some Bool
            | Int8, Byte | Byte, Int8  -> None

    /// Convert System.Type to Dtype
    let ofType (ty: System.Type) =
        if ty.Equals(typeof<int32>) then Dtype.Int32
        elif ty.Equals(typeof<double>) then Dtype.Float64
        elif ty.Equals(typeof<single>) then Dtype.Float32
        elif ty.Equals(typeof<int64>) then Dtype.Int64
        elif ty.Equals(typeof<int16>) then Dtype.Int16
        elif ty.Equals(typeof<int8>) then Dtype.Int8
        elif ty.Equals(typeof<byte>) then Dtype.Byte
        elif ty.Equals(typeof<bool>) then Dtype.Bool
        else failwithf "unknown type '%A' used as tensor type" ty

    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Dtype>()

    let Register name inOutType = codes.GetOrAdd(name, (fun _ -> incr count; Dtype.Other(name, count.Value, inOutType)))

    let mutable Default = Dtype.Float32

[<AutoOpen>]
module DtypeGlobalOps =
    let opNotSupported msg (t: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A" msg t)

    let opNotSupported2 msg (t1: Dtype) (t2: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type (%A, %A)" msg t1 t2)

