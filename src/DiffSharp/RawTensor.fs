namespace DiffSharp.RawTensor
open DiffSharp.Util

type Device =
    | CPU
    | GPU

type Backend =
    | CPUBase
    | CPUOpenBLAS
    | CPUTorch
    | GPUTorch

type DType =
    | Float16
    | Float32
    | Float64

[<AbstractClass>]
type RawTensor(value:obj, shape:int[], dtype:DType, device:Device, backend:Backend) =
    member t.Value = value
    member t.Shape = shape
    member t.Dim = shape.Length
    member t.Length = shapeLength shape
    member t.DType = dtype
    member t.Device = device
    member t.Backend = backend
    override t.ToString() = t.GetString()
    member t.Extend(shape) = t.CreateWithShape(t.ToValue(), shape)

    abstract member Create : obj -> RawTensor
    abstract member CreateWithShape : obj * int[] -> RawTensor
    abstract member Stack: seq<RawTensor> -> RawTensor
    abstract member Zero : unit -> RawTensor
    abstract member Zeros : int[] -> RawTensor
    abstract member One : unit -> RawTensor
    abstract member Ones : int[] -> RawTensor
    abstract member Random : int[] -> RawTensor
    abstract member RandomNormal : int[] -> RawTensor
    abstract member GetString : unit -> string
    abstract member ToValue: unit -> obj
    abstract member ToArray: unit -> System.Array
    abstract member Equals: RawTensor -> bool
    abstract member ApproximatelyEquals: RawTensor * float -> bool
    abstract member AddTT : RawTensor -> RawTensor
    abstract member AddTT0 : RawTensor -> RawTensor
    abstract member AddT2T1: RawTensor -> RawTensor
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
    abstract member NegT : unit -> RawTensor
    abstract member SumT : unit -> RawTensor
    abstract member SumT2Dim0 : unit -> RawTensor
    abstract member TransposeT2: unit -> RawTensor
    abstract member SignT: unit -> RawTensor
    abstract member AbsT: unit -> RawTensor
    abstract member ReLUT: unit -> RawTensor
    abstract member ExpT: unit -> RawTensor
    abstract member LogT: unit -> RawTensor