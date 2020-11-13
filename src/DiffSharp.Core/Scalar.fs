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
            | Dtype.Float32 -> x.toSingle() :> scalar
            | Dtype.Float64 -> x.toDouble() :> scalar
            | Dtype.Int8 -> x.toSByte() :> scalar
            | Dtype.Byte -> x.toByte() :> scalar
            | Dtype.Int32 -> x.toInt32() :> scalar
            | Dtype.Int64 -> x.toInt64() :> scalar
            | Dtype.Int16 -> x.toInt16() :> scalar
            | Dtype.Bool -> x.toBool() :> scalar

    // Floating point scalars force integers to widen to float32
    //
    // Double scalars don't force widen to float64
    // Int64 scalars don't force integers to widen to int64
    // Int32 scalars don't force integers to widen to int32 etc.
    //
    // This is deliberate, scalars never force widening to
    // float64, but may force widening to float32
    //
    // For example:
    //  >>> import torch
    //  >>> (torch.tensor([1], dtype=torch.int32) * 2.5).dtype
    //  torch.float32
    //  >>> (torch.tensor([1], dtype=torch.int32) * 2).dtype
    //  torch.int32
    let tryWidenScalar (tensorDtype: Dtype) (scalar: scalar) =
        match tensorDtype, scalar.GetTypeCode() with 
        | Dtype.Integral, (TypeCode.Double | TypeCode.Single) -> ValueSome Dtype.Float32
        | _, _ -> ValueNone
        
        
        