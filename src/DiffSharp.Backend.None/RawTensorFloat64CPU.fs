namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend

type RawTensorFloat64CPU(values: double[], shape:int[]) =
    inherit RawTensorCPU<double>(values, shape, Float64)

    static member Zero() = RawTensorCPU.Zero() |> RawTensorFloat64CPU
    static member One() = RawTensorCPU.One() |> RawTensorFloat64CPU
    static member Zeros(shape:int[]) = RawTensorCPU.Zeros(shape) |> RawTensorFloat64CPU
    static member Ones(shape:int[]) = RawTensorCPU.Ones(shape) |> RawTensorFloat64CPU
    static member Random(shape:int[])  = RawTensorCPU.Random double shape |> RawTensorFloat64CPU
    static member RandomNormal(shape:int[]) = RawTensorCPU.RandomNormal double shape |> RawTensorFloat64CPU
    static member Create(value:obj) = RawTensorCPU.Create double double double (value) |> RawTensorFloat64CPU
    override t1.CompareTo(t2) = RawTensorCPU.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat64CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat64CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat64CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat64CPU([|1.0|], [||])
    override t.Ones(shape) = upcast RawTensorFloat64CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat64CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = upcast (RawTensorCPU.RandomMultinomial double (t, numSamples)|> RawTensorFloat64CPU)
    override t1.Equals(t2:RawTensor) = RawTensorCPU.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPU.ApproximatelyEquals(t1, t2, double tolerance)
    override t1.LtTT(t2) = upcast (RawTensorCPU.LtTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.GtTT(t2) = upcast (RawTensorCPU.GtTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.LeTT(t2) = upcast (RawTensorCPU.LeTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.GeTT(t2) = upcast (RawTensorCPU.GeTT(t1, t2) |> RawTensorFloat64CPU)
    override t.MaxIndexT() = RawTensorCPU.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPU.MinIndexT(t)
    override t1.AddTT(t2) = upcast (RawTensorCPU.AddTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.AddTT0(t2) = upcast (RawTensorCPU.AddTT0(t1, t2) |> RawTensorFloat64CPU)
    override t1.AddT2T1(t2) = upcast (RawTensorCPU.AddT2T1(t1, t2) |> RawTensorFloat64CPU)
    override t1.AddTTSlice(location:int[], t2) = upcast (RawTensorCPU.AddTTSlice(t1, location, t2) |> RawTensorFloat64CPU)
    override t1.SubTT(t2) = upcast (RawTensorCPU.SubTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.SubT0T(t2) = upcast (RawTensorCPU.SubT0T(t1, t2) |> RawTensorFloat64CPU)
    override t1.SubTT0(t2) = upcast (RawTensorCPU.SubTT0(t1, t2) |> RawTensorFloat64CPU)
    override t1.MulTT(t2) = upcast (RawTensorCPU.MulTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.MulTT0(t2) = upcast (RawTensorCPU.MulTT0(t1, t2) |> RawTensorFloat64CPU)
    override t1.DivTT(t2) = upcast (RawTensorCPU.DivTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.DivT0T(t2) = upcast (RawTensorCPU.DivT0T(t1, t2) |> RawTensorFloat64CPU)
    override t1.DivTT0(t2) = upcast (RawTensorCPU.DivTT0(t1, t2) |> RawTensorFloat64CPU)
    override t1.PowTT(t2) = upcast (RawTensorCPU.PowTT(t1, t2) |> RawTensorFloat64CPU)
    override t1.PowT0T(t2) = upcast (RawTensorCPU.PowT0T(t1, t2) |> RawTensorFloat64CPU)
    override t1.PowTT0(t2) = upcast (RawTensorCPU.PowTT0(t1, t2) |> RawTensorFloat64CPU)
    override t1.MatMulT2T2(t2) = upcast (RawTensorCPU.MatMulT2T2(t1, t2) |> RawTensorFloat64CPU)
    override t1.Conv1D(t2, stride, padding) = upcast (RawTensorCPU.Conv1D RawTensorFloat64CPU.Zeros (t1, t2, stride, padding))
    override t.NegT() = upcast (RawTensorCPU.NegT(t) |> RawTensorFloat64CPU)
    override t.SumT() = upcast (RawTensorCPU.SumT(t) |> RawTensorFloat64CPU)
    override t.SumT2Dim0() = upcast (RawTensorCPU.SumT2Dim0(t) |> RawTensorFloat64CPU)
    override t.TransposeT2() = upcast (RawTensorCPU.TransposeT2(t) |> RawTensorFloat64CPU)
    override t.SignT() = upcast (RawTensorCPU.SignT double t |> RawTensorFloat64CPU)
    override t.FloorT() = upcast (RawTensorCPU.FloorT(t) |> RawTensorFloat64CPU)
    override t.CeilT() = upcast (RawTensorCPU.CeilT(t) |> RawTensorFloat64CPU)
    override t.RoundT() = upcast (RawTensorCPU.RoundT(t) |> RawTensorFloat64CPU)
    override t.AbsT() = upcast (RawTensorCPU.AbsT(t) |> RawTensorFloat64CPU)
    override t.ReluT() = upcast (RawTensorCPU.ReluT(t) |> RawTensorFloat64CPU)
    override t.SigmoidT() = upcast (RawTensorCPU.SigmoidT(t) |> RawTensorFloat64CPU)
    override t.ExpT() = upcast (RawTensorCPU.ExpT(t) |> RawTensorFloat64CPU)
    override t.LogT() = upcast (RawTensorCPU.LogT(t) |> RawTensorFloat64CPU)
    override t.Log10T() = upcast (RawTensorCPU.Log10T(t) |> RawTensorFloat64CPU)
    override t.SqrtT() = upcast (RawTensorCPU.SqrtT(t) |> RawTensorFloat64CPU)
    override t.SinT() = upcast (RawTensorCPU.SinT(t) |> RawTensorFloat64CPU)
    override t.CosT() = upcast (RawTensorCPU.CosT(t) |> RawTensorFloat64CPU)
    override t.TanT() = upcast (RawTensorCPU.TanT(t) |> RawTensorFloat64CPU)
    override t.SinhT() = upcast (RawTensorCPU.SinhT(t) |> RawTensorFloat64CPU)
    override t.CoshT() = upcast (RawTensorCPU.CoshT(t) |> RawTensorFloat64CPU)
    override t.TanhT() = upcast (RawTensorCPU.TanhT(t) |> RawTensorFloat64CPU)
    override t.AsinT() = upcast (RawTensorCPU.AsinT(t) |> RawTensorFloat64CPU)
    override t.AcosT() = upcast (RawTensorCPU.AcosT(t) |> RawTensorFloat64CPU)
    override t.AtanT() = upcast (RawTensorCPU.AtanT(t) |> RawTensorFloat64CPU)

and RawTensorFloat64CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat64CPU.Zero()
    override __.One = upcast RawTensorFloat64CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat64CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat64CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorFloat64CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat64CPU.Create(values)

    