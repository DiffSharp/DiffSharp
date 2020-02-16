namespace DiffSharp
open DiffSharp.Backend
open DiffSharp.Util

[<CustomEquality; CustomComparison>]
type Tensor = 
    | Tensor of RawTensor
    | TensorF of Tensor * Tensor * uint32
    | TensorR of Tensor * (Tensor ref) * TensorOp * (uint32 ref) * uint32

    member t.Primal =
        match t with
        | Tensor(_) -> t
        | TensorF(tp,_,_) -> tp
        | TensorR(tp,_,_,_,_) -> tp

    member t.PrimalRaw =
        match t with
        | Tensor(tp) -> tp
        | TensorF(tp,_,_) -> tp.PrimalRaw
        | TensorR(tp,_,_,_,_) -> tp.PrimalRaw

    member t.Depth =
        let rec depth x d =
            match x with
            | Tensor(_) -> d
            | TensorF(tp,_,_) -> depth tp (d + 1)
            | TensorR(tp,_,_,_,_) -> depth tp (d + 1)
        depth t 0

    member t.Derivative
        with get() =
            match t with
            | Tensor(_) -> failwith "Cannot get derivative of constant Tensor"
            | TensorF(_,td,_) -> td
            | TensorR(_,td,_,_,_) -> !td
        and set(value) =
            match t with
            | Tensor(_) -> failwith "Cannot set derivative of constant Tensor"
            | TensorF(_) -> failwith "Cannot set derivative of TensorF"
            | TensorR(_,td,_,_,_) -> td := value

    member t.Fanout
        with get() =
            match t with
            | Tensor(_) -> failwith "Cannot get fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot get fanout of TensorF"
            | TensorR(_,_,_,f,_) -> !f
        and set(value) =
            match t with
            | Tensor(_) -> failwith "Cannot set fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot set fanout of TensorF"
            | TensorR(_,_,_,f,_) -> f := value

    member t.ForwardDiff(derivative:Tensor, ?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        if t.Shape = derivative.Shape then TensorF(t, derivative, tag) else invalidArg "derivative" (sprintf "Expecting derivative of same shape with primal. primal: %A, derivative: %A" t derivative)
    member t.ReverseDiff(?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        TensorR(t, ref (t.Zero()), NewT, ref 0u, tag)
    member t.NoDiff() = Tensor(t.PrimalRaw)
    member t.Shape = t.PrimalRaw.Shape
    member t.Dim = t.PrimalRaw.Dim
    member t.Nelement = t.PrimalRaw.Nelement
    member t.ToArray() = t.PrimalRaw.ToArray()
    member t.ToValue() = t.PrimalRaw.ToValue()
    member t.Zero() = Tensor(t.PrimalRaw.Zero())
    member t.Create(value) = Tensor(t.PrimalRaw.Create(value))
    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.PrimalRaw.Equals(tensor.PrimalRaw)
        | _ -> false
    member t.ApproximatelyEqual(tensor:Tensor, ?tolerance) =
        let tolerance = defaultArg tolerance 0.01
        t.PrimalRaw.ApproximatelyEquals(tensor.PrimalRaw, tolerance)
    override t.GetHashCode() =
        match t with
        | Tensor(tp) -> hash (tp)
        | TensorF(tp,td,tt) -> hash (tp, td, tt)
        | TensorR(tp,td,_,_,tt) -> hash (tp, !td, tt)
    interface System.IComparable with
        override t.CompareTo(other) =
            match other with
            | :? Tensor as tensor -> 
                if t.Dim = tensor.Dim && t.Dim = 0 then
                    t.PrimalRaw.CompareTo(tensor.PrimalRaw)
                else
                    invalidOp "Cannot compare non-scalar Tensors"
            | _ -> invalidOp "Cannot compare Tensor with another type"
    member t1.IsSameDiffType(t2:Tensor) =
        match t1, t2 with
        | Tensor(_),  Tensor(_)  -> true
        | Tensor(_),  TensorF(_) -> false
        | Tensor(_),  TensorR(_) -> false
        | TensorF(_), Tensor(_)  -> false
        | TensorF(_), TensorF(_) -> true
        | TensorF(_), TensorR(_) -> false
        | TensorR(_), Tensor(_)  -> false
        | TensorR(_), TensorF(_) -> false
        | TensorR(_), TensorR(_) -> true

    static member Lt(a:Tensor, b:Tensor) = Tensor(a.PrimalRaw.LtTT(b.PrimalRaw))
    member t1.Lt(t2) = Tensor.Lt(t1, t2)
    static member Gt(a:Tensor, b:Tensor) = Tensor(a.PrimalRaw.GtTT(b.PrimalRaw))
    member t1.Gt(t2) = Tensor.Gt(t1, t2)
    static member Le(a:Tensor, b:Tensor) = Tensor(a.PrimalRaw.LeTT(b.PrimalRaw))
    member t1.Le(t2) = Tensor.Le(t1, t2)
    static member Ge(a:Tensor, b:Tensor) = Tensor(a.PrimalRaw.GeTT(b.PrimalRaw))
    member t1.Ge(t2) = Tensor.Ge(t1, t2)
    static member MaxIndex(a:Tensor) = a.PrimalRaw.MaxIndexT()
    member t.MaxIndex() = Tensor.MaxIndex(t)
    static member MinIndex(a:Tensor) = a.PrimalRaw.MinIndexT()
    member t.MinIndex() = Tensor.MinIndex(t)
    static member Max(a:Tensor) = a.[a.MaxIndex()]
    member t.Max() = Tensor.Max(t)
    static member Min(a:Tensor) = a.[a.MinIndex()]
    member t.Min() = Tensor.Min(t)
    static member Max(a:Tensor, b:Tensor) = ((a + b) + Tensor.Abs(b - a)) / 2.
    static member Min(a:Tensor, b:Tensor) = ((a + b) - Tensor.Abs(a - b)) / 2.
    static member op_Explicit(tensor:Tensor):'a = downcast tensor.PrimalRaw.ToValue()
    static member ZerosLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Zeros(tensor.Shape))
    static member ZerosLike(tensor:Tensor, shape:seq<int>) = Tensor(tensor.PrimalRaw.Zeros(shape |> Array.ofSeq))
    static member OnesLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Ones(tensor.Shape))
    static member OnesLike(tensor:Tensor, shape:seq<int>) = Tensor(tensor.PrimalRaw.Ones(shape |> Array.ofSeq))
    static member RandomLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Random(tensor.Shape))
    static member RandomLike(tensor:Tensor, shape:seq<int>) = Tensor(tensor.PrimalRaw.Random(shape |> Array.ofSeq))
    static member RandomNormalLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.RandomNormal(tensor.Shape))
    static member RandomNormalLike(tensor:Tensor, shape:seq<int>) = Tensor(tensor.PrimalRaw.RandomNormal(shape |> Array.ofSeq))

    static member Zeros(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) =
        Tensor(RawTensor.Zeros(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))

    static member Ones(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) =
        Tensor(RawTensor.Ones(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))

    static member Random(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) =
        Tensor(RawTensor.Random(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))

    static member RandomNormal(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) =
        Tensor(RawTensor.RandomNormal(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))

    static member Create(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) =
        Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))

    static member Extend(a:Tensor, shape:seq<int>) =
        if a.Dim <> 0 then invalidArg "tensor" (sprintf "Expecting a 0d Tensor, received shape: %A" a.Shape)
        match a with
        | Tensor(ap) -> Tensor(ap.Extend(shape|>Seq.toArray))
        | TensorF(ap,ad,at) ->
            let cp = Tensor.Extend(ap, shape)
            let cd = Tensor.Extend(ad, shape)
            TensorF(cp,cd,at)
        | TensorR(ap,_,_,_,at) ->
            let cp = Tensor.Extend(ap, shape)
            TensorR(cp, ref (a.Zero()), MakeTofT0(a), ref 0u, at)

    member internal t.GetSlice(bounds:int[,]) =
        if t.Dim = 0 then invalidOp "Cannot slice a scalar Tensor"
        let fullBounds = Array2D.init t.Dim 2 (fun i j -> if j=0 then 0 else t.Shape.[i]-1)
        bounds |> Array2D.iteri (fun i j v -> 
            if j=1 && v >= t.Shape.[i] then failwithf "Index outside the bounds of Tensor shape %A" t.Shape
            fullBounds.[i, j] <- v)
        match t with
        | Tensor(ap) -> Tensor(ap.GetSlice(fullBounds))
        | TensorF(ap,ad,at) -> TensorF(ap.GetSlice(fullBounds), ad.GetSlice(fullBounds), at)
        | TensorR(ap,_,_,_,at) -> TensorR(ap.GetSlice(fullBounds), ref (ap.Zero()), SliceT(t, fullBounds), ref 0u, at)

    member t.Item
        with get([<System.ParamArray>] index:int[]) =
            if t.Dim = 0 then invalidOp "Cannot index a scalar Tensor"
            if index.Length > t.Dim then invalidArg "index" (sprintf "Expecting an index with <=%i dimensions" t.Dim)
            let bounds = Array2D.init index.Length 2 (fun i _ -> index.[i])
            t.GetSlice(bounds)

    static member Stack(tensors:seq<Tensor>) = 
        // TODO: check if all Tensors are of the same type (Tensor, TensorF, or TensorR) and have the same nesting tag
        match Seq.head tensors with
        | Tensor(ap) -> Tensor(ap.StackTs(tensors |> Seq.map (fun t -> t.PrimalRaw)))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.Primal)
            let ad = tensors |> Seq.map (fun t -> t.Derivative)
            TensorF(Tensor.Stack(ap), Tensor.Stack(ad), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.Primal)
            let cp = Tensor.Stack(ap) in TensorR(cp, ref (cp.Zero()), StackTs(tensors), ref 0u, at)

    static member Unstack (a:Tensor) =
        match a with
        | Tensor(ap) -> ap.UnstackT() |> Seq.map Tensor
        | TensorF(ap,ad,at) -> Seq.map2 (fun p d -> TensorF(p,d,at)) (ap.Unstack()) (ad.Unstack())
        | TensorR(ap,_,_,_,at) -> Seq.mapi (fun i p -> TensorR(p, ref (p.Zero()), UnstackT(a, i), ref 0u, at)) (ap.Unstack())
    member t.Unstack() = Tensor.Unstack(t)

    static member inline OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev) =
        match a with
        | Tensor(ap)           -> Tensor(fRaw(ap))
        | TensorF(ap,ad,at)    -> let cp = fTensor(ap) in TensorF(cp, dfTensorFwd(cp,ap,ad), at)
        | TensorR(ap,_,_,_,at) -> let cp = fTensor(ap) in TensorR(cp, ref (a.Zero()), dfTensorRev(a), ref 0u, at)

    static member inline OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT) =
        match a, b with
        | Tensor(ap),           Tensor(bp)                      -> Tensor(fRaw(ap, bp))
        | Tensor(_),            TensorF(bp,bd,bt)               -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | Tensor(_),            TensorR(bp,_,_,_,bt)            -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorF(ap,ad,at),    Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(ap,ad,at),    TensorF(bp,bd,bt)    when at=bt -> let cp = fTensor(ap,bp) in TensorF(cp, dfTensorFwdTT(cp,ap,ad,bp,bd), at)
        | TensorF(ap,ad,at),    TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | TensorF(_,_,at),      TensorR(_,_,_,_,bt)  when at=bt -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorF(ap,ad,at),    TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorR(ap,_,_,_,at), Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(_,_,bt)      when at=bt -> failwith "Cannot have TensorR and TensorF in the same nesting level"
        | TensorR(ap,_,_,_,at), TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap, b) in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp, bp, bd), bt)
        | TensorR(ap,_,_,_,at), TensorR(bp,_,_,_,bt) when at=bt -> let cp = fTensor(ap,bp) in TensorR(cp, ref (a.Zero()), dfTensorRevTT(a,b), ref 0u, at)
        | TensorR(ap,_,_,_,at), TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | _ -> failwith "Unexpected combination of Tensors" // Won't happen, added for suppressing "incomplete matches" warning

    static member (+) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.AddTT(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT(a,b)
            let inline dfTensorRevTC(a,b) = AddTTConst(a)
            let inline dfTensorRevCT(a,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.AddTT0(a)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Extend(ad, b.Shape)
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT0(b,a)
            let inline dfTensorRevTC(a,b) = AddTConstT0(a)
            let inline dfTensorRevCT(a,b) = AddTT0Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.AddTT0(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Extend(bd, a.Shape)
            let inline dfTensorRevTT(a,b) = AddTT0(a,b)
            let inline dfTensorRevTC(a,b) = AddTT0Const(a)
            let inline dfTensorRevCT(a,b) = AddTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 2 && b.Dim = 1 then
            if a.Shape.[1] = b.Shape.[0] then
                let inline fRaw(a:RawTensor,b) = a.AddT2T1(b)
                let inline fTensor(a,b) = a + b
                let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
                let inline dfTensorFwdTC(cp,ap,ad) = ad
                let inline dfTensorFwdCT(cp,bp,bd) = Tensor.ZerosLike(cp) + bd
                let inline dfTensorRevTT(a,b) = AddT2T1(a,b)
                let inline dfTensorRevTC(a,b) = AddT2T1Const(a)
                let inline dfTensorRevCT(a,b) = AddT2ConstT1(b)
                Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape                
        elif a.Dim = 1 && b.Dim = 2 then
            if a.Shape.[0] = b.Shape.[1] then
                let inline fRaw(a,b:RawTensor) = b.AddT2T1(a)
                let inline fTensor(a,b) = a + b
                let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
                let inline dfTensorFwdTC(cp,ap,ad) = ad + Tensor.ZerosLike(cp)
                let inline dfTensorFwdCT(cp,bp,bd) = bd
                let inline dfTensorRevTT(a,b) = AddT2T1(b,a)
                let inline dfTensorRevTC(a,b) = AddT2ConstT1(a)
                let inline dfTensorRevCT(a,b) = AddT2T1Const(b)
                Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape                
        // TODO: implement general broadcasting additions
        else failwithf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape
    static member (+) (a:Tensor, b) = a + a.Create(b)
    static member (+) (a, b:Tensor) = b.Create(a) + b
    member t1.Add(t2:Tensor) = t1 + t2
    member t1.Add(t2) = t1 + t1.Create(t2)

    static member (-) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.SubTT(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubTT(a,b)
            let inline dfTensorRevTC(a,b) = SubTTConst(a)
            let inline dfTensorRevCT(a,b) = SubTConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubT0T(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Extend(ad, b.Shape)
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubT0T(a,b)
            let inline dfTensorRevTC(a,b) = SubT0TConst(a)
            let inline dfTensorRevCT(a,b) = SubT0ConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubTT0(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Extend(-bd, a.Shape)
            let inline dfTensorRevTT(a,b) = SubTT0(a,b)
            let inline dfTensorRevTC(a,b) = SubTT0Const(a)
            let inline dfTensorRevCT(a,b) = SubTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot subtract Tensors with shapes %A, %A" a.Shape b.Shape
    static member (-) (a:Tensor, b) = a - a.Create(b)
    static member (-) (a, b:Tensor) = b.Create(a) - b
    member t1.Sub(t2:Tensor) = t1 - t2
    member t1.Sub(t2) = t1 - t1.Create(t2)

    static member (*) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.MulTT(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT(a,b)
            let inline dfTensorRevTC(a,b) = MulTTConst(a,b)
            let inline dfTensorRevCT(a,b) = MulTTConst(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.MulTT0(a)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(b,a)
            let inline dfTensorRevTC(a,b) = MulTConstT0(a,b)
            let inline dfTensorRevCT(a,b) = MulTT0Const(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.MulTT0(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(a,b)
            let inline dfTensorRevTC(a,b) = MulTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = MulTConstT0(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        // TODO: implement general broadcasting?
        else failwithf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape
    static member (*) (a:Tensor, b) = a * a.Create(b)
    static member (*) (a, b:Tensor) = b.Create(a) * b
    member t1.Mul(t2:Tensor) = t1 * t2
    member t1.Mul(t2) = t1 * t1.Create(t2)

    static member (/) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.DivTT(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT(a,b)
            let inline dfTensorRevTC(a,b) = DivTTConst(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivT0T(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivT0T(a,b)
            let inline dfTensorRevTC(a,b) = DivT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = DivT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivTT0(b)
            let inline fTensor(a:Tensor,b:Tensor) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT0(a,b)
            let inline dfTensorRevTC(a,b) = DivTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot divide Tensors with shapes %A, %A" a.Shape b.Shape
    static member (/) (a:Tensor, b) = a / a.Create(b)
    static member (/) (a, b:Tensor) = b.Create(a) / b
    member t1.Div(t2:Tensor) = t1 / t2
    member t1.Div(t2) = t1 / t1.Create(t2)

    static member Pow (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.PowTT(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT(a,b)
            let inline dfTensorRevTC(a,b) = PowTTConst(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowT0T(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowT0T(a,b)
            let inline dfTensorRevTC(a,b) = PowT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = PowT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowTT0(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT0(a,b)
            let inline dfTensorRevTC(a,b) = PowTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot exponentiate Tensors with shapes %A, %A" a.Shape b.Shape
    static member Pow (a:Tensor, b) = a ** a.Create(b)
    static member Pow (a, b:Tensor) = b.Create(a) ** b
    member t1.Pow(t2:Tensor) = t1 ** t2
    member t1.Pow(t2) = t1 ** t1.Create(t2)

    static member MatMul (a:Tensor, b:Tensor) =
        if a.Dim <> 2 || b.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" a.Shape b.Shape
        if a.Shape.[1] = b.Shape.[0] then
            let inline fRaw(a:RawTensor,b) = a.MatMulT2T2(b)
            let inline fTensor(a,b) = Tensor.MatMul(a, b)
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = Tensor.MatMul(ad, bp) + Tensor.MatMul(ap, bd)
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.MatMul(ad, b)
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.MatMul(a, bd)
            let inline dfTensorRevTT(a,b) = MatMulT2T2(a,b)
            let inline dfTensorRevTC(a,b) = MatMulT2T2Const(a,b)
            let inline dfTensorRevCT(a,b) = MatMulT2ConstT2(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot multiply Tensors with shapes %A, %A" a.Shape b.Shape
    member t1.MatMul(t2:Tensor) = Tensor.MatMul(t1, t2)

    static member (~-) (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.NegT()
        let inline fTensor(a) = -a
        let inline dfTensorFwd(cp,ap,ad) = -ad
        let inline dfTensorRev(a) = NegT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Neg() = -t

    static member Sum (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SumT()
        let inline fTensor(a) = Tensor.Sum(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Sum(ad)
        let inline dfTensorRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sum() = Tensor.Sum(t)

    // TODO: this can be implemented in a more memory efficient way by pushing the sum operation to the RawTensor level and implementing the derivatives using general broadcasting when it's available
    static member Sum(a:Tensor, dim:int) =
        if dim = 0 && a.Dim = 0 then a
        else
            if dim >= a.Dim || dim < 0 then invalidArg "dim" <| sprintf "Expecting dim to be between 0 and %A" a.Dim
            let sBounds = Array2D.init a.Dim 2 (fun i j -> if j=0 then 0 else a.Shape.[i]-1)
            sBounds.[dim, 1] <- 0
            let mutable s = Tensor.ZerosLike(a).GetSlice(sBounds)
            for i=0 to a.Shape.[dim]-1 do
                sBounds.[dim,0] <- i
                sBounds.[dim,1] <- i
                s <- s + a.GetSlice(sBounds)
            s
    member t.Sum(dim) = Tensor.Sum(t, dim)

    static member Sum(a:Tensor, dim:int, keepDim:bool) = if keepDim then Tensor.Sum(a, dim).Unsqueeze(dim) else Tensor.Sum(a, dim)
    member t.Sum(dim, keepDim) = Tensor.Sum(t, dim, keepDim)

    static member Mean (a:Tensor) = Tensor.Sum(a) / a.Nelement
    member t.Mean() = Tensor.Mean(t)

    static member Mean(a:Tensor, dim:int) = 
        if dim = 0 && a.Dim = 0 then a
        else a.Sum(dim) / a.Shape.[dim]
    member t.Mean(dim) = Tensor.Mean(t, dim)

    // This is the two-pass algorithm better than the naive algorithm
    static member Variance (a:Tensor) = let a' = a - Tensor.Mean(a) in Tensor.Sum(a' * a') / (a.Nelement - 1)
    member t.Variance() = Tensor.Variance(t)

    // TODO: this is the naive algorithm, can be improved for better numerical stability
    static member Variance(a:Tensor, dim:int) =
        if dim >= a.Dim || dim < 0 then invalidArg "dim" <| sprintf "Expecting dim to be between 0 and %A" a.Dim
        let sBounds = Array2D.init a.Dim 2 (fun i j -> if j=0 then 0 else a.Shape.[i]-1)
        sBounds.[dim, 1] <- 0
        let mutable s = Tensor.ZerosLike(a).GetSlice(sBounds)
        let mutable sSquare = Tensor.ZerosLike(a).GetSlice(sBounds)
        let n = a.Shape.[dim]
        for i=0 to n-1 do
            sBounds.[dim,0] <- i
            sBounds.[dim,1] <- i
            let slice = a.GetSlice(sBounds)
            s <- s + slice
            sSquare <- sSquare + slice * slice
        (sSquare - (s * s) / n) / (n - 1)
    member t.Variance(dim) = Tensor.Variance(t, dim)

    static member Stddev (a:Tensor, dim:int) = Tensor.Variance(a, dim) |> Tensor.Sqrt
    member t.Stddev(dim) = Tensor.Stddev(t, dim)

    static member Stddev (a:Tensor) = Tensor.Variance(a) |> Tensor.Sqrt
    member t.Stddev() = Tensor.Stddev(t)

    static member SumT2Dim0 (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SumT2Dim0()
        let inline fTensor(a) = Tensor.SumT2Dim0(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.SumT2Dim0(ad)
        let inline dfTensorRev(a) = SumT2Dim0(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.SumT2Dim0() = Tensor.SumT2Dim0(t)
    
    static member Transpose (a:Tensor) =
        if a.Dim <> 2 then invalidOp <| sprintf "Expecting a 2d Tensor, received Tensor with shape %A" a.Shape
        let inline fRaw(a:RawTensor) = a.TransposeT2()
        let inline fTensor(a) = Tensor.Transpose(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Transpose(ad)
        let inline dfTensorRev(a) = TransposeT2(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Transpose() = Tensor.Transpose(t)

    static member Squeeze (a:Tensor, ?dim:int) =
        let dim = defaultArg dim -1
        let inline fRaw(a:RawTensor) = a.SqueezeT(dim)
        let inline fTensor(a) = Tensor.Squeeze(a, dim)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Squeeze(ad, dim)
        let inline dfTensorRev(a) = SqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Squeeze(?dim) = 
        let dim = defaultArg dim -1
        Tensor.Squeeze(t, dim)

    static member Unsqueeze (a:Tensor, dim:int) =
        let inline fRaw(a:RawTensor) = a.UnsqueezeT(dim)
        let inline fTensor(a) = Tensor.Unsqueeze(a, dim)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Unsqueeze(ad, dim)
        let inline dfTensorRev(a) = UnsqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Unsqueeze(dim) = Tensor.Unsqueeze(t, dim)

    static member Flip (a:Tensor, dims:int[]) =
        let inline fRaw(a:RawTensor) = a.FlipT(dims)
        let inline fTensor(a) = Tensor.Flip(a, dims)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Flip(ad, dims)
        let inline dfTensorRev(a) = FlipT(a, dims)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Flip(dims) = Tensor.Flip(t, dims)

    static member Repeat (a:Tensor, dim:int, times:int) =
        if a.Shape.[dim] <> 1 then invalidOp <| sprintf "Expecting Tensor's shape at dim to be 1, received Tensor with shape %A and dim %A" a.Shape dim
        let newShape = a.Shape |> Array.copy
        newShape.[dim] <- times
        let mutable ret = Tensor.ZerosLike(a, newShape)
        let location = Array.create a.Dim 0
        for i=0 to times-1 do
            location.[dim] <- i
            ret <- Tensor.AddSlice(ret, location, a)
        ret
    member t.Repeat(dim:int, times:int) = Tensor.Repeat(t, dim, times)

    static member View (a:Tensor, shape:seq<int>) =
        let shape = shape |> Seq.toArray |> shapeComplete a.Nelement  // Handles -1 semantics
        let inline fRaw(a:RawTensor) = a.ViewT(shape)
        let inline fTensor(a) = Tensor.View(a, shape)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.View(ad, shape)
        let inline dfTensorRev(a) = ViewT(a, a.Shape)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.View(shape) = Tensor.View(t, shape)
    member t.View(shape:int) = Tensor.View(t, [|shape|])

    static member ViewAs(a:Tensor, b:Tensor) = a.View(b.Shape)
    member a.ViewAs(b:Tensor) = a.View(b.Shape)

    static member Sign (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SignT()
        let inline fTensor(a) = Tensor.Sign(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.ZerosLike(cp)
        let inline dfTensorRev(a) = SignT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sign() = Tensor.Sign(t)

    static member Floor (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.FloorT()
        let inline fTensor(a) = Tensor.Floor(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.ZerosLike(cp)
        let inline dfTensorRev(a) = FloorT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Floor() = Tensor.Floor(t)

    static member Ceil (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.CeilT()
        let inline fTensor(a) = Tensor.Ceil(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.ZerosLike(cp)
        let inline dfTensorRev(a) = CeilT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Ceil() = Tensor.Ceil(t)

    static member Round (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.RoundT()
        let inline fTensor(a) = Tensor.Round(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.ZerosLike(cp)
        let inline dfTensorRev(a) = RoundT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Round() = Tensor.Round(t)

    static member Abs (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.AbsT()
        let inline fTensor(a) = Tensor.Abs(a)
        let inline dfTensorFwd(cp,ap,ad) = ad * Tensor.Sign(ap)
        let inline dfTensorRev(a) = AbsT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Abs() = Tensor.Abs(t)

    static member Relu (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.ReluT()
        let inline fTensor(a) = Tensor.Relu(a)
        let inline dfTensorFwd(cp,ap,ad) = let sap = Tensor.Sign(ap) in ad * Tensor.Abs(sap) * (1. + sap) / 2.
        let inline dfTensorRev(a) = ReluT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Relu() = Tensor.Relu(t)

    static member LeakyRelu (a:Tensor, ?negativeSlope:float) =
        let negativeSlope = defaultArg negativeSlope 0.01
        Tensor.Max(Tensor.Create(0.), a) + negativeSlope * Tensor.Min(Tensor.Create(0.), a)
    member t.LeakyRelu() = Tensor.LeakyRelu(t)
    member t.LeakyRelu(negativeSlope) = Tensor.LeakyRelu(t, negativeSlope)

    static member Sigmoid (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SigmoidT()
        let inline fTensor(a) = Tensor.Sigmoid(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad * cp * (1. - cp)
        let inline dfTensorRev(a) = SigmoidT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sigmoid() = Tensor.Sigmoid(t)

    static member Exp (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.ExpT()
        let inline fTensor(a) = Tensor.Exp(a)
        let inline dfTensorFwd(cp,ap,ad) = ad * cp
        let inline dfTensorRev(a) = ExpT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Exp() = Tensor.Exp(t)

    static member Log (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.LogT()
        let inline fTensor(a) = Tensor.Log(a)
        let inline dfTensorFwd(cp,ap,ad) = ad / ap
        let inline dfTensorRev(a) = LogT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Log() = Tensor.Log(t)

    static member Log10 (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.Log10T()
        let inline fTensor(a) = Tensor.Log10(a)
        let inline dfTensorFwd(cp,ap:Tensor,ad) = ad / (ap * log10Val)
        let inline dfTensorRev(a) = Log10T(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Log10() = Tensor.Log10(t)

    static member Sqrt (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SqrtT()
        let inline fTensor(a) = Tensor.Sqrt(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad / (2. * cp)
        let inline dfTensorRev(a) = SqrtT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sqrt() = Tensor.Sqrt(t)

    static member Sin (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SinT()
        let inline fTensor(a) = Tensor.Sin(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad * Tensor.Cos(ap)
        let inline dfTensorRev(a) = SinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sin() = Tensor.Sin(t)

    static member Cos (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.CosT()
        let inline fTensor(a) = Tensor.Cos(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = -ad * Tensor.Sin(ap)
        let inline dfTensorRev(a) = CosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Cos() = Tensor.Cos(t)

    static member Tan (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.TanT()
        let inline fTensor(a) = Tensor.Tan(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = let cosap = Tensor.Cos(ap) in ad / (cosap * cosap)
        let inline dfTensorRev(a) = TanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Tan() = Tensor.Tan(t)

    static member Sinh (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SinhT()
        let inline fTensor(a) = Tensor.Sinh(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad * Tensor.Cosh(ap)
        let inline dfTensorRev(a) = SinhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sinh() = Tensor.Sinh(t)

    static member Cosh (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.CoshT()
        let inline fTensor(a) = Tensor.Cosh(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad * Tensor.Sinh(ap)
        let inline dfTensorRev(a) = CoshT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Cosh() = Tensor.Cosh(t)

    static member Tanh (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.TanhT()
        let inline fTensor(a) = Tensor.Tanh(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = let coshap = Tensor.Cosh(ap) in ad / (coshap * coshap)
        let inline dfTensorRev(a) = TanhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Tanh() = Tensor.Tanh(t)

    static member Asin (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.AsinT()
        let inline fTensor(a) = Tensor.Asin(a)
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / Tensor.Sqrt(1. - ap*ap)
        let inline dfTensorRev(a) = AsinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Asin() = Tensor.Asin(t)

    static member Acos (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.AcosT()
        let inline fTensor(a) = Tensor.Acos(a)
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = -ad / Tensor.Sqrt(1. - ap*ap)
        let inline dfTensorRev(a) = AcosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Acos() = Tensor.Acos(t)

    static member Atan (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.AtanT()
        let inline fTensor(a) = Tensor.Atan(a)
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / (1. + ap*ap)
        let inline dfTensorRev(a) = AtanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Atan() = Tensor.Atan(t)

    static member AddSlice (a:Tensor, location:seq<int>, b:Tensor) =
        if not (shapeContains a.Shape b.Shape) then failwithf "Expecting a.Shape to contain b.Shape, received %A, %A" a.Shape b.Shape
        if location |> Seq.length <> a.Dim then failwithf "Expecting location of the same length as a.Dim, received %A, %A" (location |> Seq.length) a.Dim
        let location = location |> Seq.toArray
        let inline fRaw(a:RawTensor,b) = a.AddTTSlice(location, b)
        let inline fTensor(a,b) = Tensor.AddSlice(a, location, b)
        let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = Tensor.AddSlice(ad, location, bd)
        let inline dfTensorFwdTC(cp,ap,ad) = ad
        let inline dfTensorFwdCT(cp,bp,bd) = Tensor.AddSlice(Tensor.ZerosLike(cp), location, bd)
        let inline dfTensorRevTT(a,b) = AddTTSlice(a,location,b)
        let inline dfTensorRevTC(a,b) = AddTTConstSlice(a)
        let inline dfTensorRevCT(a,b) = AddTConstTSlice(location,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    static member Softmax(a:Tensor, dim:int) =
        if dim < 0 || dim >= a.Dim then failwithf "Expecting 0 <= dim < a.Dim, received %A, %A" dim a.Dim
        let e = (a - a.Max().NoDiff()).Exp()
        let esum = e.Sum(dim, keepDim=true).Repeat(dim, a.Shape.[dim])
        e / esum
    member t.Softmax(dim:int) = Tensor.Softmax(t, dim)

    static member MSELoss(a:Tensor, b:Tensor) = let z = a - b in (z * z).Mean()

    static member Conv1D(a:Tensor, b:Tensor, ?stride:int, ?padding:int) =
        // a: input
        // b: filter
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let inline fRaw(a:RawTensor,b) = a.Conv1D(b, stride, padding)
        let inline fTensor(a,b) = Tensor.Conv1D(a, b, stride, padding)
        let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = Tensor.Conv1D(ad, bp, stride, padding) + Tensor.Conv1D(ap, bd, stride, padding)
        let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Conv1D(ad, b, stride, padding)
        let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Conv1D(a, bd, stride, padding)
        // TODO: implement the derivatives (the following are placeholders from Tensor.MatMul)
        let inline dfTensorRevTT(a,b) = Conv1DTT(a,b, stride, padding)
        let inline dfTensorRevTC(a,b) = Conv1DTTConst(a,b, stride, padding)
        let inline dfTensorRevCT(a,b) = Conv1DTConstT(a,b, stride, padding)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)


    member t.Reverse(?value:Tensor, ?zeroDerivatives:bool) =
        let value = defaultArg value (Tensor.OnesLike(t))
        let zeroDerivatives = defaultArg zeroDerivatives true
        if value.Shape <> t.Shape then invalidArg "value" <| sprintf "Expecting an adjoint value of shape %A, but received of shape %A" t.Shape value.Shape
        t.ReverseReset(zeroDerivatives)
        t.ReversePush(value)

    member inline t.Backward(value) = t.Reverse(value)

    member t.ReverseReset(zeroDerivatives:bool) =
        let rec reset (ts: Tensor list) =
            match ts with
            | [] -> ()
            | t :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    if zeroDerivatives then t.Derivative <- t.Zero()
                    t.Fanout <- t.Fanout + 1u
                    if t.Fanout = 1u then
                        match o with
                        | AddTT(a,b) -> reset (a::b::tt)
                        | AddTTConst(a) -> reset (a::tt)
                        | AddTT0(a,b) -> reset (a::b::tt)
                        | AddTT0Const(a) -> reset (a::tt)
                        | AddTConstT0(b) -> reset (b::tt)
                        | AddT2T1(a,b) -> reset (a::b::tt)
                        | AddT2T1Const(a) -> reset (a::tt)
                        | AddT2ConstT1(b) -> reset (b::tt)
                        | SubTT(a,b) -> reset (a::b::tt)
                        | SubTTConst(a) -> reset (a::tt)
                        | SubTConstT(b) -> reset (b::tt)
                        | SubT0T(a,b) -> reset (a::b::tt)
                        | SubT0TConst(a) -> reset (a::tt)
                        | SubT0ConstT(b) -> reset (b::tt)
                        | SubTT0(a,b) -> reset (a::b::tt)
                        | SubTT0Const(a) -> reset (a::tt)
                        | SubTConstT0(b) -> reset (b::tt)
                        | MulTT(a,b) -> reset (a::b::tt)
                        | MulTTConst(a,_) -> reset (a::tt)
                        | MulTT0(a,b) -> reset (a::b::tt)
                        | MulTConstT0(_,b) -> reset (b::tt)
                        | MulTT0Const(a,_) -> reset (a::tt)
                        | DivTT(a,b) -> reset (a::b::tt)
                        | DivTTConst(a,_) -> reset (a::tt)
                        | DivTConstT(_,b) -> reset (b::tt)
                        | DivT0T(a,b) -> reset (a::b::tt)
                        | DivT0TConst(a,_) -> reset (a::tt)
                        | DivT0ConstT(_,b) -> reset (b::tt)
                        | DivTT0(a,b) -> reset (a::b::tt)
                        | DivTT0Const(a,_) -> reset (a::tt)
                        | DivTConstT0(_,b) -> reset (b::tt)
                        | PowTT(a,b) -> reset (a::b::tt)
                        | PowTTConst(a,_) -> reset (a::tt)
                        | PowTConstT(_,b) -> reset (b::tt)
                        | PowT0T(a,b) -> reset (a::b::tt)
                        | PowT0TConst(a,_) -> reset (a::tt)
                        | PowT0ConstT(_,b) -> reset (b::tt)
                        | PowTT0(a,b) -> reset (a::b::tt)
                        | PowTT0Const(a,_) -> reset (a::tt)
                        | PowTConstT0(_,b) -> reset (b::tt)
                        | MatMulT2T2(a,b) -> reset (a::b::tt)
                        | MatMulT2T2Const(a,_) -> reset (a::tt)
                        | MatMulT2ConstT2(_,b) -> reset (b::tt)
                        | Conv1DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv1DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv1DTConstT(_,b,_,_) -> reset (b::tt)
                        | NegT(a) -> reset (a::tt)
                        | SumT(a) -> reset (a::tt)
                        | SumT2Dim0(a) -> reset (a::tt)
                        | MakeTofT0(a) -> reset (a::tt)
                        | StackTs(a) -> reset (List.append (a |> List.ofSeq) tt)
                        | UnstackT(a,_) -> reset (a::tt)
                        | TransposeT2(a) -> reset (a::tt)
                        | SqueezeT(a) -> reset (a::tt)
                        | UnsqueezeT(a) -> reset (a::tt)
                        | FlipT(a,_) -> reset (a::tt)
                        | ViewT(a,_) -> reset (a::tt)
                        | SliceT(a,_) -> reset (a::tt)
                        | AddTTSlice(a,_,b) -> reset (a::b::tt)
                        | AddTTConstSlice(a) -> reset (a::tt)
                        | AddTConstTSlice(_, b) -> reset (b::tt)
                        | SignT(a) -> reset (a::tt)
                        | FloorT(a) -> reset (a::tt)
                        | CeilT(a) -> reset (a::tt)
                        | RoundT(a) -> reset (a::tt)
                        | AbsT(a) -> reset (a::tt)
                        | ReluT(a) -> reset (a::tt)
                        | SigmoidT(a) -> reset (a::tt)
                        | ExpT(a) -> reset (a::tt)
                        | LogT(a) -> reset (a::tt)
                        | Log10T(a) -> reset (a::tt)
                        | SqrtT(a) -> reset (a::tt)
                        | SinT(a) -> reset (a::tt)
                        | CosT(a) -> reset (a::tt)
                        | TanT(a) -> reset (a::tt)
                        | SinhT(a) -> reset (a::tt)
                        | CoshT(a) -> reset (a::tt)
                        | TanhT(a) -> reset (a::tt)
                        | AsinT(a) -> reset (a::tt)
                        | AcosT(a) -> reset (a::tt)
                        | AtanT(a) -> reset (a::tt)
                        | NewT -> reset tt
                    else reset tt
                | _ -> reset tt
        reset [t]

    member t.ReversePush(value:Tensor) =
        let rec push (ts:(Tensor*Tensor) list) =
            match ts with
            | [] -> ()
            | (v, t) :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    t.Derivative <- t.Derivative + v
                    t.Fanout <- t.Fanout - 1u
                    if t.Fanout = 0u then
                        // printfn "reversepush"
                        // printfn "t %A" t
                        // printfn "o %A" o
                        match o with
                        | AddTT(a,b) -> push ((t.Derivative, a) :: (t.Derivative, b) :: tt)
                        | AddTTConst(a) -> push ((t.Derivative, a) :: tt)
                        | AddTT0(a,b) -> push ((t.Derivative, a) :: (t.Derivative.Sum(), b) :: tt)
                        | AddTT0Const(a) -> push ((t.Derivative, a) :: tt)
                        | AddTConstT0(b) -> push ((t.Derivative.Sum(), b) :: tt)
                        | AddT2T1(a,b) -> push ((t.Derivative, a) :: (t.Derivative.SumT2Dim0(), b) :: tt)
                        | AddT2T1Const(a) -> push ((t.Derivative, a) :: tt)
                        | AddT2ConstT1(b) -> push ((t.Derivative.SumT2Dim0(), b) :: tt)
                        | SubTT(a,b) -> push ((t.Derivative, a) :: (-t.Derivative, b) :: tt)
                        | SubTTConst(a) -> push ((t.Derivative, a) :: tt)
                        | SubTConstT(b) -> push ((-t.Derivative, b) :: tt)
                        | SubT0T(a,b) -> push ((t.Derivative.Sum(), a) :: (-t.Derivative, b) :: tt)
                        | SubT0TConst(a) -> push ((t.Derivative.Sum(), a) :: tt)
                        | SubT0ConstT(b) -> push ((-t.Derivative, b) :: tt)
                        | SubTT0(a,b) -> push ((t.Derivative, a) :: (-t.Derivative.Sum(), b) :: tt)
                        | SubTT0Const(a) -> push ((t.Derivative, a) :: tt)
                        | SubTConstT0(b) -> push ((-t.Derivative.Sum(), b) :: tt)      
                        | MulTT(a,b) -> push ((t.Derivative * b.Primal, a) :: (t.Derivative * a.Primal, b) :: tt)
                        | MulTTConst(a,b) -> push ((t.Derivative * b, a) :: tt)
                        | MulTT0(a,b) -> push ((t.Derivative * b.Primal, a) :: ((t.Derivative * a.Primal).Sum(), b) :: tt)
                        | MulTConstT0(a,b) -> push (((t.Derivative * a).Sum(), b) :: tt)
                        | MulTT0Const(a,b) -> push ((t.Derivative * b, a) :: tt)
                        | DivTT(a,b) -> push ((t.Derivative / b.Primal, a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))), b) :: tt)
                        | DivTTConst(a,b) -> push ((t.Derivative / b, a) :: tt)
                        | DivTConstT(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))), b) :: tt)
                        | DivT0T(a,b) -> push (((t.Derivative / b.Primal).Sum(), a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))), b) :: tt)
                        | DivT0TConst(a,b) -> push (((t.Derivative / b).Sum(), a) :: tt)
                        | DivT0ConstT(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))), b) :: tt)
                        | DivTT0(a,b) -> push ((t.Derivative / b.Primal, a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))).Sum(), b) :: tt)
                        | DivTT0Const(a,b) -> push ((t.Derivative / b, a) :: tt)
                        | DivTConstT0(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))).Sum(), b) :: tt)
                        | PowTT(a,b) -> push ((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal, a) :: (t.Derivative * (a.Primal ** b.Primal) * log a.Primal, b) :: tt)
                        | PowTTConst(a,b) -> push ((t.Derivative * (a.Primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT(a,b) -> push ((t.Derivative * (a ** b.Primal) * log a, b) :: tt)
                        | PowT0T(a,b) -> push (((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal).Sum(), a) :: (t.Derivative * (a.Primal ** b.Primal) * log a.Primal, b) :: tt)
                        | PowT0TConst(a,b) -> push (((t.Derivative * (a.Primal ** (b - 1.)) * b).Sum(), a) :: tt)
                        | PowT0ConstT(a,b) -> push ((t.Derivative * (a ** b.Primal) * log a, b) :: tt)
                        | PowTT0(a,b) -> push ((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal, a) :: ((t.Derivative * (a.Primal ** b.Primal) * log a.Primal).Sum(), b) :: tt)
                        | PowTT0Const(a,b) -> push ((t.Derivative * (a.Primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT0(a,b) -> push (((t.Derivative * (a ** b.Primal) * log a).Sum(), b) :: tt)
                        | MatMulT2T2(a,b) -> push ((Tensor.MatMul(t.Derivative, b.Primal.Transpose()), a) :: (Tensor.MatMul(a.Primal.Transpose(), t.Derivative), b) :: tt)
                        | MatMulT2T2Const(a,b) -> push ((Tensor.MatMul(t.Derivative, b.Transpose()), a) :: tt)
                        | MatMulT2ConstT2(a,b) -> push ((Tensor.MatMul(a.Transpose(), t.Derivative), b) :: tt)
                        | Conv1DTT(a,b,_,_) -> failwith "Not implemented"
                        | Conv1DTTConst(a,_,_,_) -> failwith "Not implemented"
                        | Conv1DTConstT(_,b,_,_) -> failwith "Not implemented"           
                        | NegT(a) -> push ((-t.Derivative, a) :: tt)
                        | SumT(a) -> push ((Tensor.Extend(t.Derivative, a.Shape), a) :: tt)
                        | SumT2Dim0(a) -> push ((Tensor.ZerosLike(a) + t.Derivative, a) :: tt)
                        | MakeTofT0(a) -> push ((t.Derivative.Sum(), a) :: tt)
                        | StackTs(a) ->  push (List.append (a |> Seq.map2 (fun t a -> (t, a)) (t.Derivative.Unstack()) |> Seq.toList) tt)
                        | UnstackT(a,i) -> 
                            if a.Derivative.Dim = 0 then a.Derivative <- Tensor.ZerosLike(a) + a.Derivative
                            a.Derivative <- Tensor.AddSlice(a.Derivative, Array.init a.Dim (fun j -> if j=0 then i else 0), t.Derivative.Unsqueeze(0))
                            push ((a.Zero(), a) :: tt)
                        | TransposeT2(a) -> push ((t.Derivative.Transpose(), a) :: tt)
                        | SqueezeT(a) -> push ((t.Derivative.ViewAs(a), a) :: tt)
                        | UnsqueezeT(a) -> push ((t.Derivative.ViewAs(a), a) :: tt)
                        | FlipT(a, dims) -> push ((t.Derivative.Flip(dims), a) :: tt)
                        | ViewT(a,aShape) -> push (((t.Derivative.View(aShape)), a) :: tt)
                        | SliceT(a,bounds) -> 
                            // TODO: Tensor.ZerosLike(a) below is to handle non-scalar TensorRs with a scalar derivative Tensor(0.) (representing the initialization before accumulation). This is correct but can be changed to eliminate the extra op.
                            if a.Derivative.Dim = 0 then a.Derivative <- Tensor.ZerosLike(a) + a.Derivative
                            a.Derivative <- Tensor.AddSlice(a.Derivative, boundsToLocation bounds, t.Derivative.View(boundsToShape bounds))
                            push ((a.Zero(), a) :: tt)
                        | AddTTSlice(a,location,b) -> push ((t.Derivative, a) :: (t.Derivative.GetSlice(shapeLocationToBounds b.Shape location), b):: tt)
                        | AddTTConstSlice(a) -> push ((t.Derivative, a) :: tt)
                        | AddTConstTSlice(location, b) -> push ((t.Derivative.GetSlice(shapeLocationToBounds b.Shape location), b):: tt)
                        | SignT(a) -> push ((Tensor.ZerosLike(a), a) :: tt)
                        | FloorT(a) -> push ((Tensor.ZerosLike(a), a) :: tt)
                        | CeilT(a) -> push ((Tensor.ZerosLike(a), a) :: tt)
                        | RoundT(a) -> push ((Tensor.ZerosLike(a), a) :: tt)
                        | AbsT(a) -> push ((t.Derivative * a.Primal.Sign(), a) :: tt)
                        | ReluT(a) -> let sap = a.Primal.Sign() in push ((t.Derivative * (sap.Abs()) * (sap + 1.) / 2., a) :: tt)
                        | SigmoidT(a) -> push ((t.Derivative * t.Primal * (1. - t.Primal), a) :: tt)
                        | ExpT(a) -> push ((t.Derivative * t.Primal, a) :: tt)
                        | LogT(a) -> push ((t.Derivative / a.Primal, a) :: tt)
                        | Log10T(a) -> push ((t.Derivative / (a.Primal * log10Val), a) :: tt)
                        | SqrtT(a) -> push ((t.Derivative / (2. * t.Primal), a) :: tt)
                        | SinT(a) -> push ((t.Derivative * (a.Primal.Cos()), a) :: tt)
                        | CosT(a) -> push ((-t.Derivative * (a.Primal.Sin()), a) :: tt)
                        | TanT(a) -> let cosap = a.Primal.Cos() in push ((t.Derivative / (cosap * cosap), a) :: tt)
                        | SinhT(a) -> push ((t.Derivative * (a.Primal.Cosh()), a) :: tt)
                        | CoshT(a) -> push ((t.Derivative * (a.Primal.Sinh()), a) :: tt)
                        | TanhT(a) -> let coshap = a.Primal.Cosh() in push ((t.Derivative / (coshap * coshap), a) :: tt)
                        | AsinT(a) -> push ((t.Derivative / Tensor.Sqrt(1. - a.Primal*a.Primal), a) :: tt)
                        | AcosT(a) -> push ((-t.Derivative / Tensor.Sqrt(1. - a.Primal*a.Primal), a) :: tt)
                        | AtanT(a) -> push ((t.Derivative / (1. + a.Primal*a.Primal), a) :: tt)
                        | NewT -> push tt
                    else push tt
                | _ -> push tt
        push [(value, t)]

and TensorOp =
    | AddTT of Tensor * Tensor
    | AddTTConst of Tensor
    | AddTT0 of Tensor * Tensor
    | AddTT0Const of Tensor
    | AddTConstT0 of Tensor
    | AddT2T1 of Tensor * Tensor
    | AddT2T1Const of Tensor
    | AddT2ConstT1 of Tensor
    
    | SubTT of Tensor * Tensor
    | SubTTConst of Tensor
    | SubTConstT of Tensor
    | SubT0T of Tensor * Tensor
    | SubT0TConst of Tensor
    | SubT0ConstT of Tensor
    | SubTT0 of Tensor * Tensor
    | SubTT0Const of Tensor
    | SubTConstT0 of Tensor

    | MulTT of Tensor * Tensor
    | MulTTConst of Tensor * Tensor
    | MulTT0 of Tensor * Tensor
    | MulTConstT0 of Tensor * Tensor
    | MulTT0Const of Tensor * Tensor

    | DivTT of Tensor * Tensor
    | DivTTConst of Tensor * Tensor
    | DivTConstT of Tensor * Tensor
    | DivT0T of Tensor * Tensor
    | DivT0TConst of Tensor * Tensor
    | DivT0ConstT of Tensor * Tensor
    | DivTT0 of Tensor * Tensor
    | DivTT0Const of Tensor * Tensor
    | DivTConstT0 of Tensor * Tensor

    | PowTT of Tensor * Tensor
    | PowTTConst of Tensor * Tensor
    | PowTConstT of Tensor * Tensor
    | PowT0T of Tensor * Tensor
    | PowT0TConst of Tensor * Tensor
    | PowT0ConstT of Tensor * Tensor
    | PowTT0 of Tensor * Tensor
    | PowTT0Const of Tensor * Tensor
    | PowTConstT0 of Tensor * Tensor

    | MatMulT2T2 of Tensor * Tensor
    | MatMulT2T2Const of Tensor * Tensor
    | MatMulT2ConstT2 of Tensor * Tensor

    | Conv1DTT of Tensor * Tensor * int * int
    | Conv1DTTConst of Tensor * Tensor * int * int
    | Conv1DTConstT of Tensor * Tensor * int * int

    | NegT of Tensor
    | SumT of Tensor
    | SumT2Dim0 of Tensor
    | MakeTofT0 of Tensor
    | StackTs of seq<Tensor>
    | UnstackT of Tensor * int
    | SliceT of Tensor * int[,]
    | AddTTSlice of Tensor * int[] * Tensor
    | AddTTConstSlice of Tensor
    | AddTConstTSlice of int[] * Tensor
    | TransposeT2 of Tensor
    | SqueezeT of Tensor
    | UnsqueezeT of Tensor
    | FlipT of Tensor * int[]
    | ViewT of Tensor * int[]
    | SignT of Tensor
    | FloorT of Tensor
    | CeilT of Tensor
    | RoundT of Tensor
    | AbsT of Tensor
    | ReluT of Tensor
    | SigmoidT of Tensor
    | ExpT of Tensor
    | LogT of Tensor
    | Log10T of Tensor
    | SqrtT of Tensor
    | SinT of Tensor
    | CosT of Tensor
    | TanT of Tensor
    | SinhT of Tensor
    | CoshT of Tensor
    | TanhT of Tensor
    | AsinT of Tensor
    | AcosT of Tensor
    | AtanT of Tensor
    | NewT

type Tensor with
    member t.GetSlice(i0min:int option, i0max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let bounds = array2D [[i0min; i0max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int) =
        let i0min = i0
        let i0max = i0
        let bounds = array2D [[i0min; i0max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5:int) =
        let i0min = defaultArg i0min 0
        let i0max = defaultArg i0max (t.Shape.[0] - 1)
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = defaultArg i1min 0
        let i1max = defaultArg i1max (t.Shape.[1] - 1)
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = defaultArg i2min 0
        let i2max = defaultArg i2max (t.Shape.[2] - 1)
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = defaultArg i3min 0
        let i3max = defaultArg i3max (t.Shape.[3] - 1)
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = defaultArg i4min 0
        let i4max = defaultArg i4max (t.Shape.[4] - 1)
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = defaultArg i5min 0
        let i5max = defaultArg i5max (t.Shape.[5] - 1)
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5:int) =
        let i0min = i0
        let i0max = i0
        let i1min = i1
        let i1max = i1
        let i2min = i2
        let i2max = i2
        let i3min = i3
        let i3max = i3
        let i4min = i4
        let i4max = i4
        let i5min = i5
        let i5max = i5
        let bounds = array2D [[i0min; i0max]; [i1min; i1max]; [i2min; i2max]; [i3min; i3max]; [i4min; i4max]; [i5min; i5max]]
        t.GetSlice(bounds)


[<RequireQualifiedAccess>]
module Tensor =
    let create (count:int) (value:float32) = Tensor.Create(Array.create count value)
    let zeroCreate (count:int) = Tensor.Create(Array.zeroCreate count)
    let init (count:int) (initializer:int->float32) = Tensor.Create(Array.init count initializer)
    let shape (t:Tensor) = t.Shape
    let dim (t:Tensor) = t.Dim