namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend

type RawTensorFloat64CPU(values: double[], shape:int[]) =
    inherit RawTensorCPU<double>(values, shape, Float64)

    static member Zero() = RawTensorCPUTemplates.Zero() : RawTensorFloat64CPU
    static member One() = RawTensorCPUTemplates.One() : RawTensorFloat64CPU
    static member Zeros(shape:int[]) = RawTensorCPUTemplates.Zeros(shape) : RawTensorFloat64CPU
    static member Ones(shape:int[]) = RawTensorCPUTemplates.Ones(shape) : RawTensorFloat64CPU
    static member Random(shape:int[])  = RawTensorCPUTemplates.Random double shape : RawTensorFloat64CPU
    static member RandomNormal(shape:int[]) = RawTensorCPUTemplates.RandomNormal double shape : RawTensorFloat64CPU
    static member Create(value:obj) = RawTensorCPUTemplates.Create double double double (value) : RawTensorFloat64CPU
    override t1.CompareTo(t2) = RawTensorCPUTemplates.CompareTo(t1, (t2 :?> RawTensorFloat64CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat64CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat64CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat64CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat64CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat64CPU([|1.0|], [||])
    override t.Ones(shape) = upcast RawTensorFloat64CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat64CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = upcast (RawTensorCPUTemplates.RandomMultinomial double (t, numSamples): RawTensorFloat64CPU)
    override t1.Equals(t2:RawTensor) = RawTensorCPUTemplates.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPUTemplates.ApproximatelyEquals(t1, t2, double tolerance)
    override t1.LtTT(t2) = upcast (RawTensorCPUTemplates.LtTT(t1, t2) : RawTensorFloat64CPU)
    override t1.GtTT(t2) = upcast (RawTensorCPUTemplates.GtTT(t1, t2) : RawTensorFloat64CPU)
    override t1.LeTT(t2) = upcast (RawTensorCPUTemplates.LeTT(t1, t2) : RawTensorFloat64CPU)
    override t1.GeTT(t2) = upcast (RawTensorCPUTemplates.GeTT(t1, t2) : RawTensorFloat64CPU)
    override t.MaxIndexT() = RawTensorCPUTemplates.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPUTemplates.MinIndexT(t)
    override t1.AddTT(t2) = upcast (RawTensorCPUTemplates.AddTT(t1, t2) : RawTensorFloat64CPU)
    override t1.AddTT0(t2) = upcast (RawTensorCPUTemplates.AddTT0(t1, t2) : RawTensorFloat64CPU)
    override t1.AddT2T1(t2) = upcast (RawTensorCPUTemplates.AddT2T1(t1, t2) : RawTensorFloat64CPU)
    override t1.AddTTSlice(location:int[], t2) = upcast (RawTensorCPUTemplates.AddTTSlice(t1, location, t2) : RawTensorFloat64CPU)
    override t1.SubTT(t2) = upcast (RawTensorCPUTemplates.SubTT(t1, t2) : RawTensorFloat64CPU)
    override t1.SubT0T(t2) = upcast (RawTensorCPUTemplates.SubT0T(t1, t2) : RawTensorFloat64CPU)
    override t1.SubTT0(t2) = upcast (RawTensorCPUTemplates.SubTT0(t1, t2) : RawTensorFloat64CPU)
    override t1.MulTT(t2) = upcast (RawTensorCPUTemplates.MulTT(t1, t2) : RawTensorFloat64CPU)
    override t1.MulTT0(t2) = upcast (RawTensorCPUTemplates.MulTT0(t1, t2) : RawTensorFloat64CPU)
    override t1.DivTT(t2) = upcast (RawTensorCPUTemplates.DivTT(t1, t2) : RawTensorFloat64CPU)
    override t1.DivT0T(t2) = upcast (RawTensorCPUTemplates.DivT0T(t1, t2) : RawTensorFloat64CPU)
    override t1.DivTT0(t2) = upcast (RawTensorCPUTemplates.DivTT0(t1, t2) : RawTensorFloat64CPU)
    override t1.PowTT(t2) = upcast (RawTensorCPUTemplates.PowTT(t1, t2) : RawTensorFloat64CPU)
    override t1.PowT0T(t2) = upcast (RawTensorCPUTemplates.PowT0T(t1, t2) : RawTensorFloat64CPU)
    override t1.PowTT0(t2) = upcast (RawTensorCPUTemplates.PowTT0(t1, t2) : RawTensorFloat64CPU)
    override t1.MatMulT2T2(t2) = upcast (RawTensorCPUTemplates.MatMulT2T2(t1, t2) : RawTensorFloat64CPU)
    override t1.Conv1D(t2, stride, padding) = upcast (RawTensorCPUTemplates.Conv1D(t1, t2, stride, padding) : RawTensorFloat64CPU)
    override t.NegT() = upcast (RawTensorCPUTemplates.NegT(t) : RawTensorFloat64CPU)
    override t.SumT() = upcast (RawTensorCPUTemplates.SumT(t) : RawTensorFloat64CPU)
    override t.SumT2Dim0() = upcast (RawTensorCPUTemplates.SumT2Dim0(t) : RawTensorFloat64CPU)
    override t.TransposeT2() = upcast (RawTensorCPUTemplates.TransposeT2(t) : RawTensorFloat64CPU)
    override t.SqueezeT(dim) = upcast (RawTensorCPUTemplates.SqueezeT(t, dim) : RawTensorFloat64CPU)
    override t.UnsqueezeT(dim) = upcast (RawTensorCPUTemplates.UnsqueezeT(t, dim) : RawTensorFloat64CPU)
    override t.ViewT(shape:int[]) = upcast (RawTensorCPUTemplates.ViewT(t, shape) : RawTensorFloat64CPU)
    override t.SignT() = upcast (RawTensorCPUTemplates.SignT double t : RawTensorFloat64CPU)
    override t.FloorT() = upcast (RawTensorCPUTemplates.FloorT(t) : RawTensorFloat64CPU)
    override t.CeilT() = upcast (RawTensorCPUTemplates.CeilT(t) : RawTensorFloat64CPU)
    override t.RoundT() = upcast (RawTensorCPUTemplates.RoundT(t) : RawTensorFloat64CPU)
    override t.AbsT() = upcast (RawTensorCPUTemplates.AbsT(t) : RawTensorFloat64CPU)
    override t.ReluT() = upcast (RawTensorCPUTemplates.ReluT(t) : RawTensorFloat64CPU)
    override t.SigmoidT() = upcast (RawTensorCPUTemplates.SigmoidT(t) : RawTensorFloat64CPU)
    override t.ExpT() = upcast (RawTensorCPUTemplates.ExpT(t) : RawTensorFloat64CPU)
    override t.LogT() = upcast (RawTensorCPUTemplates.LogT(t) : RawTensorFloat64CPU)
    override t.Log10T() = upcast (RawTensorCPUTemplates.Log10T(t) : RawTensorFloat64CPU)
    override t.SqrtT() = upcast (RawTensorCPUTemplates.SqrtT(t) : RawTensorFloat64CPU)
    override t.SinT() = upcast (RawTensorCPUTemplates.SinT(t) : RawTensorFloat64CPU)
    override t.CosT() = upcast (RawTensorCPUTemplates.CosT(t) : RawTensorFloat64CPU)
    override t.TanT() = upcast (RawTensorCPUTemplates.TanT(t) : RawTensorFloat64CPU)
    override t.SinhT() = upcast (RawTensorCPUTemplates.SinhT(t) : RawTensorFloat64CPU)
    override t.CoshT() = upcast (RawTensorCPUTemplates.CoshT(t) : RawTensorFloat64CPU)
    override t.TanhT() = upcast (RawTensorCPUTemplates.TanhT(t) : RawTensorFloat64CPU)
    override t.AsinT() = upcast (RawTensorCPUTemplates.AsinT(t) : RawTensorFloat64CPU)
    override t.AcosT() = upcast (RawTensorCPUTemplates.AcosT(t) : RawTensorFloat64CPU)
    override t.AtanT() = upcast (RawTensorCPUTemplates.AtanT(t) : RawTensorFloat64CPU)

and RawTensorFloat64CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat64CPU.Zero()
    override __.One = upcast RawTensorFloat64CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat64CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat64CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorFloat64CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat64CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat64CPU.Create(values)

    