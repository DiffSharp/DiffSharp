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
    member t.Length = getShapeLength shape
    member t.DType = dtype
    member t.Device = device
    member t.Backend = backend
    override t.ToString() = t.GetString()
    member t.Extend(shape) = t.CreateWithShape(t.ToValue(), shape)

    static member (+) (t1:RawTensor, t2:RawTensor) = t1.Add(t2)
    member t1.Add(t2) =
        match t1, t2 with
        | t1, t2 when t1.Shape = t2.Shape -> t1.AddTT(t2)
        | t1, t2 when t1.Dim = 0 -> t2.AddTT0(t1)
        | t1, t2 when t2.Dim = 0 -> t1.AddTT0(t2)
        | t1, t2 when t1.Dim = 2 && t2.Dim = 1 ->
            if t1.Shape.[0] = t2.Shape.[0] then
                t1.AddT2T1(t2)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" t1.Shape t2.Shape
        | t1, t2 when t1.Dim = 1 && t2.Dim = 2 ->
            if t1.Shape.[0] = t2.Shape.[0] then
                t2.AddT2T1(t1)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" t1.Shape t2.Shape
        // TODO: implement general broadcasting additions
        | _ -> invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" t1.Shape t2.Shape

    static member (-) (t1:RawTensor, t2:RawTensor) = t1.Sub(t2)
    member t1.Sub(t2) =
        match t1, t2 with
        | t1, t2 when t1.Shape = t2.Shape -> t1.SubTT(t2)
        | t1, t2 when t1.Dim = 0 -> t1.SubT0T(t2)
        | t1, t2 when t2.Dim = 0 -> t1.SubTT0(t2)
        // TODO: implement general broadcasting subtractions
        | _ -> invalidOp <| sprintf "Cannot subtract Tensors with shapes %A, %A" t1.Shape t2.Shape

    static member (~-) (t:RawTensor) = t.Neg()

    abstract member Create : obj -> RawTensor
    abstract member CreateWithShape : obj * int[] -> RawTensor
    abstract member Zero : unit -> RawTensor
    abstract member Zeros : int[] -> RawTensor
    abstract member One : unit -> RawTensor
    abstract member Ones : int[] -> RawTensor
    abstract member GetString : unit -> string
    abstract member ToValue: unit -> obj
    abstract member ToArray: unit -> System.Array
    abstract member Equals: RawTensor -> bool
    abstract member AddTT : RawTensor -> RawTensor
    abstract member AddTT0 : RawTensor -> RawTensor
    abstract member AddT2T1: RawTensor -> RawTensor
    abstract member SubTT : RawTensor -> RawTensor
    abstract member SubT0T : RawTensor -> RawTensor
    abstract member SubTT0 : RawTensor -> RawTensor
    abstract member Neg : unit -> RawTensor
    abstract member Sum : unit -> RawTensor
    abstract member SumT2Dim1 : unit -> RawTensor