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
        