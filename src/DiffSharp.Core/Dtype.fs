// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

/// Represents a storage type for elements of a tensor
[<Struct>]
type Dtype =
    /// Store elements as 16-bit floating point numbers (bfloat16 variation)
    | [<Experimental("Support for bfloat16 is experimental. For the reference backend, float32 representations are used. For the Torch backend, numerous operations give exceptions, you should use float32 alternatives instead and convert the tensors.")>] 
      BFloat16
    /// Store elements as 16-bit floating point numbers
    | [<Experimental("Support for float16 is experimental. For the reference backend, float32 representations are used. For the Torch backend, numerous operations give exceptions, you should use float32 alternatives instead and convert the tensors.")>]
      Float16
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
        | BFloat16 -> "BFloat16"
        | Float16 -> "Float16"
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int8 -> "Int8"
        | Byte -> "Byte"
        | Int16 -> "Int16"
        | Int32 -> "Int32"
        | Int64 -> "Int64"
        | Bool -> "Bool"

    /// Gets the natural result of the Sum(), SumToSize() and Sum(dim) operation on this dtype
    member t.SummationType =
        match t with
        | Bool | Byte | Int8 | Int16 | Int32 | Int64 -> Dtype.Int64
        | dt -> dt

    override x.ToString() = x.Name

/// Contains global functions and settings related to tensor element types, used when writing backends.
[<AutoOpen>]
module DtypeAutoOpens =

    type Dtype with
        /// Matches all floating point tensor element types
        member x.IsFloatingPoint =
            match x with
            | Float16 | BFloat16 | Float32 | Float64 -> true
            | _ -> false

        /// Matches all integral tensor element types
        member x.IsIntegral =
            match x with
            | Byte | Int8 | Int16 | Int32 | Int64 -> true
            | _ -> false

    /// Raise an exception indicating the given operation is not supported for the given tensor element type.
    let opNotSupported msg (dtype: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A" msg dtype)

    /// Raise an exception indicating the given operation is not supported for the given tensor device type.
    let opNotSupportedOnDeviceType msg (dtype: Dtype) (deviceType: DeviceType) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A on device type %A" msg dtype deviceType)

    /// Raise an exception indicating the given binary operation is not supported for the two given tensor element types.
    let opNotSupported2 msg (dtype1: Dtype) (dtype2: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type (%A, %A)" msg dtype1 dtype2)

/// Contains functions and settings related to tensor element types
module Dtype =

    /// Matches all floating point tensor element types
    let (|FloatingPoint|_|) (x: Dtype) = if x.IsFloatingPoint then Some() else None

    /// Matches all integral tensor element types
    let (|Integral|_|) (x: Dtype) = if x.IsIntegral then Some() else None

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
            | BFloat16, _ | _, BFloat16 -> Some BFloat16
            | Float16, _ | _, Float16 -> Some Float16
            | Int64, _ | _, Int64 -> Some Int64
            | Int32, _ | _, Int32 -> Some Int32
            | Int16, _ | _, Int16 -> Some Int16
            | Int8, Bool | Bool, Int8 -> Some Int8
            | Byte, Bool | Bool, Byte -> Some Byte
            | Int8, Int8 -> Some Int8
            | Byte, Byte -> Some Byte
            | Bool, Bool -> Some Bool
            | Int8, Byte | Byte, Int8  -> None

    /// Get or set the default element type used when creating tensors. Only floating point types are supported as the default type. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Dtype.Float32

    /// Find the Dtype which would result from dividing tensors with dtype1 and dtype2
    let divisionType (dtype1: Dtype) (dtype2: Dtype) =
        match dtype1.IsFloatingPoint, dtype2.IsFloatingPoint with
        | false, false -> Default
        | false, true -> dtype2
        | true, false -> dtype1
        | true, true -> (widen dtype1 dtype2).Value


