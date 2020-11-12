namespace DiffSharp

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
            match x with 
            | :? single -> Dtype.Float32
            | :? double -> Dtype.Float64
            | :? int32 -> Dtype.Int32
            | :? int64 -> Dtype.Int64
            | :? int8 -> Dtype.Int8
            | :? uint8 -> Dtype.Byte
            | :? int16 -> Dtype.Int16
            | :? bool -> Dtype.Bool
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
        