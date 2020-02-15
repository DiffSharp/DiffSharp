namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend

type RawTensorInt32CPU(values: int32[], shape:int[]) =
    inherit RawTensorCPU<int32>(values, shape, Int32)

    static member Zero() = RawTensorCPUTemplates.Zero() : RawTensorInt32CPU
    static member One() = RawTensorCPUTemplates.One() : RawTensorInt32CPU
    static member Zeros(shape:int[]) = RawTensorCPUTemplates.Zeros(shape) : RawTensorInt32CPU
    static member Ones(shape:int[]) = RawTensorCPUTemplates.Ones(shape) : RawTensorInt32CPU
    static member Random(shape:int[])  = RawTensorCPUTemplates.Random int32 shape : RawTensorInt32CPU
    static member RandomNormal(shape:int[]) = RawTensorCPUTemplates.RandomNormal int32 shape : RawTensorInt32CPU
    static member Create(value:obj) = RawTensorCPUTemplates.Create int32 int32 int32 (value) : RawTensorInt32CPU
    override t1.CompareTo(t2) = RawTensorCPUTemplates.CompareTo(t1, (t2 :?> RawTensorInt32CPU))
    override t.CreateShaped(values, shape) = upcast RawTensorInt32CPU(values, shape)
    override t.Create(values) = upcast RawTensorInt32CPU.Create(values)
    override t.Zero() = upcast RawTensorInt32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorInt32CPU.Zeros(shape)
    override t.One() = upcast RawTensorInt32CPU([|1|], [||])
    override t.Ones(shape) = upcast RawTensorInt32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorInt32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override t.RandomMultinomial(numSamples) = upcast (RawTensorCPUTemplates.RandomMultinomial int32 (t, numSamples): RawTensorInt32CPU)
    override t1.Equals(t2:RawTensor) = RawTensorCPUTemplates.Equals(t1, t2)
    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) = RawTensorCPUTemplates.ApproximatelyEquals(t1, t2, int32 tolerance)
    override t1.LtTT(t2) = upcast (RawTensorCPUTemplates.LtTT(t1, t2) : RawTensorInt32CPU)
    override t1.GtTT(t2) = upcast (RawTensorCPUTemplates.GtTT(t1, t2) : RawTensorInt32CPU)
    override t1.LeTT(t2) = upcast (RawTensorCPUTemplates.LeTT(t1, t2) : RawTensorInt32CPU)
    override t1.GeTT(t2) = upcast (RawTensorCPUTemplates.GeTT(t1, t2) : RawTensorInt32CPU)
    override t.MaxIndexT() = RawTensorCPUTemplates.MaxIndexT(t)
    override t.MinIndexT() = RawTensorCPUTemplates.MinIndexT(t)
    override t1.AddTT(t2) = upcast (RawTensorCPUTemplates.AddTT(t1, t2) : RawTensorInt32CPU)
    override t1.AddTT0(t2) = upcast (RawTensorCPUTemplates.AddTT0(t1, t2) : RawTensorInt32CPU)
    override t1.AddT2T1(t2) = upcast (RawTensorCPUTemplates.AddT2T1(t1, t2) : RawTensorInt32CPU)
    override t1.AddTTSlice(location:int[], t2) = upcast (RawTensorCPUTemplates.AddTTSlice(t1, location, t2) : RawTensorInt32CPU)
    override t1.SubTT(t2) = upcast (RawTensorCPUTemplates.SubTT(t1, t2) : RawTensorInt32CPU)
    override t1.SubT0T(t2) = upcast (RawTensorCPUTemplates.SubT0T(t1, t2) : RawTensorInt32CPU)
    override t1.SubTT0(t2) = upcast (RawTensorCPUTemplates.SubTT0(t1, t2) : RawTensorInt32CPU)
    override t1.MulTT(t2) = upcast (RawTensorCPUTemplates.MulTT(t1, t2) : RawTensorInt32CPU)
    override t1.MulTT0(t2) = upcast (RawTensorCPUTemplates.MulTT0(t1, t2) : RawTensorInt32CPU)
    override t1.DivTT(t2) = upcast (RawTensorCPUTemplates.DivTT(t1, t2) : RawTensorInt32CPU)
    override t1.DivT0T(t2) = upcast (RawTensorCPUTemplates.DivT0T(t1, t2) : RawTensorInt32CPU)
    override t1.DivTT0(t2) = upcast (RawTensorCPUTemplates.DivTT0(t1, t2) : RawTensorInt32CPU)
    override t1.MatMulT2T2(t2) = upcast (RawTensorCPUTemplates.MatMulT2T2(t1, t2) : RawTensorInt32CPU)
    override t1.Conv1D(t2, stride, padding) = upcast (RawTensorCPUTemplates.Conv1D(t1, t2, stride, padding) : RawTensorInt32CPU)
    override t.NegT() = upcast (RawTensorCPUTemplates.NegT(t) : RawTensorInt32CPU)
    override t.SumT() = upcast (RawTensorCPUTemplates.SumT(t) : RawTensorInt32CPU)
    override t.SumT2Dim0() = upcast (RawTensorCPUTemplates.SumT2Dim0(t) : RawTensorInt32CPU)
    override t.TransposeT2() = upcast (RawTensorCPUTemplates.TransposeT2(t) : RawTensorInt32CPU)
    override t.SqueezeT(dim) = upcast (RawTensorCPUTemplates.SqueezeT(t, dim) : RawTensorInt32CPU)
    override t.UnsqueezeT(dim) = upcast (RawTensorCPUTemplates.UnsqueezeT(t, dim) : RawTensorInt32CPU)
    override t.ViewT(shape:int[]) = upcast (RawTensorCPUTemplates.ViewT(t, shape) : RawTensorInt32CPU)
    override t.SignT() = upcast (RawTensorCPUTemplates.SignT int32 t : RawTensorInt32CPU)

    override t.AbsT() = upcast (RawTensorCPUTemplates.AbsT(t) : RawTensorInt32CPU)
    override t.ReluT() = upcast (RawTensorCPUTemplates.ReluT(t) : RawTensorInt32CPU)

    member t.ToFloat32() = t.Cast(Float32) :?> RawTensorFloat32CPU
    override t1.PowTT(t2) = upcast (RawTensorCPUTemplates.PowTT(t1.ToFloat32(), t2) : RawTensorFloat32CPU)
    override t1.PowT0T(t2) = upcast (RawTensorCPUTemplates.PowT0T(t1.ToFloat32(), t2) : RawTensorFloat32CPU)
    override t1.PowTT0(t2) = upcast (RawTensorCPUTemplates.PowTT0(t1.ToFloat32(), t2) : RawTensorFloat32CPU)
    override t.FloorT() = upcast (RawTensorCPUTemplates.FloorT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.CeilT() = upcast (RawTensorCPUTemplates.CeilT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.RoundT() = upcast (RawTensorCPUTemplates.RoundT(t.ToFloat32()) : RawTensorFloat32CPU)

    // Note, these produce Float32 tensors implicitly
    // TODO: check this
    override t.SigmoidT() = upcast (RawTensorCPUTemplates.SigmoidT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.ExpT() = upcast (RawTensorCPUTemplates.ExpT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.LogT() = upcast (RawTensorCPUTemplates.LogT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.Log10T() = upcast (RawTensorCPUTemplates.Log10T(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.SqrtT() = upcast (RawTensorCPUTemplates.SqrtT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.SinT() = upcast (RawTensorCPUTemplates.SinT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.CosT() = upcast (RawTensorCPUTemplates.CosT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.TanT() = upcast (RawTensorCPUTemplates.TanT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.SinhT() = upcast (RawTensorCPUTemplates.SinhT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.CoshT() = upcast (RawTensorCPUTemplates.CoshT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.TanhT() = upcast (RawTensorCPUTemplates.TanhT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.AsinT() = upcast (RawTensorCPUTemplates.AsinT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.AcosT() = upcast (RawTensorCPUTemplates.AcosT(t.ToFloat32()) : RawTensorFloat32CPU)
    override t.AtanT() = upcast (RawTensorCPUTemplates.AtanT(t.ToFloat32()) : RawTensorFloat32CPU)

and RawTensorInt32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorInt32CPU.Zero()
    override __.One = upcast RawTensorInt32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorInt32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorInt32CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorInt32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorInt32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorInt32CPU.Create(values)

    