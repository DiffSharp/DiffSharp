// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

open System
open System.Reflection

/// Represents a scalar on the DiffSharp programming model
type scalar = System.IConvertible

[<AutoOpen>]
module ScalarExtensions =
    type System.IConvertible with
        member inline x.toSingle() = x.ToSingle(null)
        member inline x.toDouble() = x.ToDouble(null)
        member inline x.toInt64() = x.ToInt64(null)
        member inline x.toInt32() = x.ToInt32(null)
        member inline x.toInt16() = x.ToInt16(null)
        member inline x.toSByte() = x.ToSByte(null)
        member inline x.toByte() = x.ToByte(null)
        member inline x.toBool() = x.toInt32() <> 0 
        member inline x.sub(y:scalar) : scalar = (x.toDouble() - y.toDouble()) :> scalar
        member inline x.log() : scalar = x.toDouble() |> log :> scalar
        member inline x.neg() : scalar = -x.toDouble() :> scalar
        member inline x.dtype =
            let ti = x.GetTypeCode()
            match ti with 
            | TypeCode.Double -> Dtype.Float64
            | TypeCode.Single -> Dtype.Float32
            | TypeCode.Int32 -> Dtype.Int32
            | TypeCode.Int64 -> Dtype.Int64
            | TypeCode.SByte -> Dtype.Int8
            | TypeCode.Byte -> Dtype.Byte
            | TypeCode.Int16 -> Dtype.Int16
            | TypeCode.Boolean -> Dtype.Bool
            | _ -> failwithf "unknown scalar type '%A'" x

        member inline x.cast(dtype) =
            match dtype with 
            | Dtype.Float16 -> x.toSingle() :> scalar
            | Dtype.BFloat16 -> x.toSingle() :> scalar
            | Dtype.Float32 -> x.toSingle() :> scalar
            | Dtype.Float64 -> x.toDouble() :> scalar
            | Dtype.Int8 -> x.toSByte() :> scalar
            | Dtype.Byte -> x.toByte() :> scalar
            | Dtype.Int32 -> x.toInt32() :> scalar
            | Dtype.Int64 -> x.toInt64() :> scalar
            | Dtype.Int16 -> x.toInt16() :> scalar
            | Dtype.Bool -> x.toBool() :> scalar

    // Floating point scalars force integers to widen to the default floating point type
    //
    // For example:
    //  >>> import torch
    //  >>> (torch.tensor([1], dtype=torch.int32) * 2.5).dtype
    //  torch.float32
    //  >>> torch.set_default_dtype(torch.float16)
    //  >>> (torch.tensor([1], dtype=torch.int32) * 2.5).dtype
    //  torch.float16
    //  >>> (torch.tensor([1], dtype=torch.int32) * 2).dtype
    //  torch.int32
    let tryWidenScalar (tensorDtype: Dtype) (scalar: scalar) =
        match tensorDtype, scalar.GetTypeCode() with 
        | Dtype.Integral, (TypeCode.Double | TypeCode.Single) -> ValueSome Dtype.Default
        | _, _ -> ValueNone
        
    let widenScalarForDivision (tensorDtype: Dtype) (scalarDtype: Dtype) =
        match tensorDtype.IsFloatingPoint, scalarDtype.IsFloatingPoint with
        | false, false -> Dtype.Default
        | false, true -> Dtype.Default
        | true, false -> tensorDtype
        | true, true -> tensorDtype

        