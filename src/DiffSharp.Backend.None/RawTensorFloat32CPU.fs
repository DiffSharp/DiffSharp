namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend

type RawTensorFloat32CPU(values: float32[], shape:int[]) =
    inherit RawTensorCPU<float32>(values, shape, Float32)

    static member Zero() = RawTensorCPUTemplates.Zero() : RawTensorFloat32CPU
    static member One() = RawTensorCPUTemplates.One() : RawTensorFloat32CPU
    static member Zeros(shape:int[]) = RawTensorCPUTemplates.Zeros(shape) : RawTensorFloat32CPU
    static member Ones(shape:int[]) = RawTensorCPUTemplates.Ones(shape) : RawTensorFloat32CPU
    static member Random(shape:int[])  = RawTensorCPUTemplates.Random float32 shape : RawTensorFloat32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPUTemplates.RandomNormal float32 shape : RawTensorFloat32CPU
    static member Create(value:obj) = RawTensorCPUTemplates.Create float32 float32 float32 (value) : RawTensorFloat32CPU
    override t1.CompareTo(t2) = RawTensorCPUTemplates.CompareTo(t1, (t2 :?> RawTensorFloat32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorFloat32CPU(values, shape)
    override t.Create(values) = upcast RawTensorFloat32CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat32CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPU([|1.f|], [||])
    override t.Ones(shape) = upcast RawTensorFloat32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = upcast (RawTensorCPUTemplates.RandomMultinomial float32 (t, numSamples): RawTensorFloat32CPU)
    override t1.Equals(t2:RawTensor) = RawTensorCPUTemplates.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPUTemplates.ApproximatelyEquals(t1, t2, float32 tolerance)
    override t1.LtTT(t2) = upcast (RawTensorCPUTemplates.LtTT(t1, t2) : RawTensorFloat32CPU)
    override t1.GtTT(t2) = upcast (RawTensorCPUTemplates.GtTT(t1, t2) : RawTensorFloat32CPU)
    override t1.LeTT(t2) = upcast (RawTensorCPUTemplates.LeTT(t1, t2) : RawTensorFloat32CPU)
    override t1.GeTT(t2) = upcast (RawTensorCPUTemplates.GeTT(t1, t2) : RawTensorFloat32CPU)
    override t.MaxIndexT() = RawTensorCPUTemplates.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPUTemplates.MinIndexT(t)
    override t1.AddTT(t2) = upcast (RawTensorCPUTemplates.AddTT(t1, t2) : RawTensorFloat32CPU)
    override t1.AddTT0(t2) = upcast (RawTensorCPUTemplates.AddTT0(t1, t2) : RawTensorFloat32CPU)
    override t1.AddT2T1(t2) = upcast (RawTensorCPUTemplates.AddT2T1(t1, t2) : RawTensorFloat32CPU)
    override t1.AddTTSlice(location:int[], t2) = upcast (RawTensorCPUTemplates.AddTTSlice(t1, location, t2) : RawTensorFloat32CPU)
    override t1.SubTT(t2) = upcast (RawTensorCPUTemplates.SubTT(t1, t2) : RawTensorFloat32CPU)
    override t1.SubT0T(t2) = upcast (RawTensorCPUTemplates.SubT0T(t1, t2) : RawTensorFloat32CPU)
    override t1.SubTT0(t2) = upcast (RawTensorCPUTemplates.SubTT0(t1, t2) : RawTensorFloat32CPU)
    override t1.MulTT(t2) = upcast (RawTensorCPUTemplates.MulTT(t1, t2) : RawTensorFloat32CPU)
    override t1.MulTT0(t2) = upcast (RawTensorCPUTemplates.MulTT0(t1, t2) : RawTensorFloat32CPU)
    override t1.DivTT(t2) = upcast (RawTensorCPUTemplates.DivTT(t1, t2) : RawTensorFloat32CPU)
    override t1.DivT0T(t2) = upcast (RawTensorCPUTemplates.DivT0T(t1, t2) : RawTensorFloat32CPU)
    override t1.DivTT0(t2) = upcast (RawTensorCPUTemplates.DivTT0(t1, t2) : RawTensorFloat32CPU)
    override t1.PowTT(t2) = upcast (RawTensorCPUTemplates.PowTT(t1, t2) : RawTensorFloat32CPU)
    override t1.PowT0T(t2) = upcast (RawTensorCPUTemplates.PowT0T(t1, t2) : RawTensorFloat32CPU)
    override t1.PowTT0(t2) = upcast (RawTensorCPUTemplates.PowTT0(t1, t2) : RawTensorFloat32CPU)
    override t1.MatMulT2T2(t2) = upcast (RawTensorCPUTemplates.MatMulT2T2(t1, t2) : RawTensorFloat32CPU)
    override t1.Conv1D(t2, stride, padding) = upcast (RawTensorCPUTemplates.Conv1D(t1, t2, stride, padding) : RawTensorFloat32CPU)
    override t.NegT() = upcast (RawTensorCPUTemplates.NegT(t) : RawTensorFloat32CPU)
    override t.SumT() = upcast (RawTensorCPUTemplates.SumT(t) : RawTensorFloat32CPU)
    override t.SumT2Dim0() = upcast (RawTensorCPUTemplates.SumT2Dim0(t) : RawTensorFloat32CPU)
    override t.TransposeT2() = upcast (RawTensorCPUTemplates.TransposeT2(t) : RawTensorFloat32CPU)
    override t.SqueezeT(dim) = upcast (RawTensorCPUTemplates.SqueezeT(t, dim) : RawTensorFloat32CPU)
    override t.UnsqueezeT(dim) = upcast (RawTensorCPUTemplates.UnsqueezeT(t, dim) : RawTensorFloat32CPU)
    override t.ViewT(shape:int[]) = upcast (RawTensorCPUTemplates.ViewT(t, shape) : RawTensorFloat32CPU)
    override t.SignT() = upcast (RawTensorCPUTemplates.SignT float32 t : RawTensorFloat32CPU)
    override t.FloorT() = upcast (RawTensorCPUTemplates.FloorT(t) : RawTensorFloat32CPU)
    override t.CeilT() = upcast (RawTensorCPUTemplates.CeilT(t) : RawTensorFloat32CPU)
    override t.RoundT() = upcast (RawTensorCPUTemplates.RoundT(t) : RawTensorFloat32CPU)
    override t.AbsT() = upcast (RawTensorCPUTemplates.AbsT(t) : RawTensorFloat32CPU)
    override t.ReluT() = upcast (RawTensorCPUTemplates.ReluT(t) : RawTensorFloat32CPU)
    override t.SigmoidT() = upcast (RawTensorCPUTemplates.SigmoidT(t) : RawTensorFloat32CPU)
    override t.ExpT() = upcast (RawTensorCPUTemplates.ExpT(t) : RawTensorFloat32CPU)
    override t.LogT() = upcast (RawTensorCPUTemplates.LogT(t) : RawTensorFloat32CPU)
    override t.Log10T() = upcast (RawTensorCPUTemplates.Log10T(t) : RawTensorFloat32CPU)
    override t.SqrtT() = upcast (RawTensorCPUTemplates.SqrtT(t) : RawTensorFloat32CPU)
    override t.SinT() = upcast (RawTensorCPUTemplates.SinT(t) : RawTensorFloat32CPU)
    override t.CosT() = upcast (RawTensorCPUTemplates.CosT(t) : RawTensorFloat32CPU)
    override t.TanT() = upcast (RawTensorCPUTemplates.TanT(t) : RawTensorFloat32CPU)
    override t.SinhT() = upcast (RawTensorCPUTemplates.SinhT(t) : RawTensorFloat32CPU)
    override t.CoshT() = upcast (RawTensorCPUTemplates.CoshT(t) : RawTensorFloat32CPU)
    override t.TanhT() = upcast (RawTensorCPUTemplates.TanhT(t) : RawTensorFloat32CPU)
    override t.AsinT() = upcast (RawTensorCPUTemplates.AsinT(t) : RawTensorFloat32CPU)
    override t.AcosT() = upcast (RawTensorCPUTemplates.AcosT(t) : RawTensorFloat32CPU)
    override t.AtanT() = upcast (RawTensorCPUTemplates.AtanT(t) : RawTensorFloat32CPU)

and RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat32CPU.Zero()
    override __.One = upcast RawTensorFloat32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat32CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorFloat32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat32CPU.Create(values)

    