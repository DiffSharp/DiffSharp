namespace DiffSharp


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
    | Float32
    | Float64
    | Int8
    | Int16
    | Int32
    | Int64
    | Bool

    member internal x.Code =
        match x with
        | Float32  -> 0x100
        | Float64 -> 0x200
        | Int8 -> 0x300
        | Int16 -> 0x400
        | Int32 -> 0x500
        | Int64 -> 0x600
        | Bool -> 0x700

    member internal x.Name =
        match x with
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int8 -> "Int8"
        | Int16 -> "Int16"
        | Int32 -> "Int32"
        | Int64 -> "Int64"
        | Bool -> "Bool"

module DType =
    /// Find the shape into which shape1 and shape2 can be expanded
    let widen (dtype1: DType) (dtype2: DType) =
        if dtype1 = dtype2 then dtype1
        else
            match dtype1, dtype2 with 
            | Float64, _ | _, Float64 -> Float64
            | Float32, _ | _, Float32 -> Float32
            | Int64, _ | _, Int64 -> Int64
            | Int32, _ | _, Int32 -> Int32
            | Int16, _ | _, Int16 -> Int16
            | Int8, _ | _, Int8 -> Int8
            | _ -> Bool

    /// Find the shape into which shape1 and shape2 can be expanded
    let ofType (ty: System.Type) =
        if ty.Equals(typeof<int32>) then DType.Int32
        elif ty.Equals(typeof<double>) then DType.Float64
        elif ty.Equals(typeof<single>) then DType.Float32
        elif ty.Equals(typeof<int64>) then DType.Int64
        elif ty.Equals(typeof<int16>) then DType.Int16
        elif ty.Equals(typeof<int8>) then DType.Int8
        elif ty.Equals(typeof<bool>) then DType.Bool
        else failwithf "unknown type '%A' used as tensor type" ty

