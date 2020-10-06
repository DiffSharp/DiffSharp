namespace DiffSharp

/// Represents a storage type for elements of a tensor
[<Struct>]
type Dtype =
    //| Float16
    /// Store elements as 32-bit floating point numbers
    | Float32
    /// Store elements as 64-bit floating point numbers
    | Float64
    /// Store elements as 8-bit integers
    | Int8
    /// Store elements as 8-bit unsigned integers
    | Byte
    /// Store elements as 16-bit signed integers
    | Int16
    /// Store elements as 32-bit signed integers
    | Int32
    /// Store elements as 64-bit signed integers
    | Int64
    /// Store elements as booleans
    | Bool

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

    /// Get the .NET type that corresponds to this type when data is transferred to .NET
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

    /// Gets the natural result of the Sum(), SumToSize() and Sum(dim) operation on this dtype
    member t.SummationType =
        match t with
        | Bool | Byte | Int8 | Int16 | Int32 | Int64 -> Dtype.Int64
        | dt -> dt

/// Contains functions and settings related to tensor element types
module Dtype =
    /// Matches all floating point tensor element types
    let (|FloatingPoint|_|) x =
        match x with
        | Float32 | Float64 -> Some()
        | _ -> None

    /// Matches all integral tensor element types
    let (|Integral|_|) x =
        match x with
        | Byte | Int8 | Int16 | Int32 | Int64 -> Some()
        | _ -> None

    /// Matches all integral or boolean tensor element types
    let (|IntegralOrBool|_|) x =
        match x with
        | Integral | Bool -> Some()
        | _ -> None

    /// Find the Dtype into which dtype1 and dtype2 can be widened
    let widen (dtype1: Dtype) (dtype2: Dtype) =
        if dtype1 = dtype2 then Some dtype1
        else
            match dtype1, dtype2 with 
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

    /// Get or set the default element type used when creating tensors. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Dtype.Float32

/// Contains global functions and settings related to tensor element types, used when writing backends.
[<AutoOpen>]
module DtypeAutoOpens =

    /// Raise an exception indicating the given operation is not supported for the given tensor element type.
    let opNotSupported msg (dtype: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A" msg dtype)

    /// Raise an exception indicating the given operation is not supported for the given tensor device type.
    let opNotSupportedOnDeviceType msg (dtype: Dtype) (deviceType: DeviceType) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A on device type %A" msg dtype deviceType)

    /// Raise an exception indicating the given binary operation is not supported for the two given tensor element types.
    let opNotSupported2 msg (dtype1: Dtype) (dtype2: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type (%A, %A)" msg dtype1 dtype2)

