namespace DiffSharp
open DiffSharp.RawTensor
open DiffSharp.Util

[<CustomEquality; NoComparison>]
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
        let rec primalRaw x =
            match x with
            | Tensor(tp) -> tp
            | TensorF(tp,_,_) -> primalRaw tp
            | TensorR(tp,_,_,_,_) -> primalRaw tp
        primalRaw t

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

    // member inline t.Value = 0.
    member t.GetForward(derivative, ?tag:uint32) = 
        let tag = defaultArg tag GlobalTagger.Next
        if t.ShapeEquals(derivative) then TensorF(t, derivative, tag) else invalidArg "derivative" (sprintf "Expecting derivative of same shape with primal. primal: %A, derivative: %A" t derivative)
    member t.GetReverse(?tag:uint32) = 
        let tag = defaultArg tag GlobalTagger.Next
        TensorR(t, ref (t.Zero()), NewT, ref 0u, tag)
    member t.Shape = t.PrimalRaw.Shape
    member t.Dim = t.PrimalRaw.Dim
    member t.ToArray() = t.PrimalRaw.ToArray()
    member t.Zero() = Tensor(t.PrimalRaw.Zero())
    member t.CreateWithSameType(value) = Tensor(t.PrimalRaw.Create(value))
    member t.ShapeEquals(tensor:Tensor) = Util.arraysEqual t.Shape tensor.Shape
    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.PrimalRaw.Equals(tensor.PrimalRaw)
        | _ -> false
    override t.GetHashCode() =
        match t with
        | Tensor(tp) -> hash (tp)
        | TensorF(tp,td,tt) -> hash (tp, td, tt)
        | TensorR(tp,td,_,_,tt) -> hash (tp, !td, tt)
    static member inline op_Explicit(tensor:Tensor):'a = downcast tensor.PrimalRaw.ToValue()
    static member ZerosLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Zeros(tensor.Shape))
    static member OnesLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Ones(tensor.Shape))
    static member Zeros(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Zeros(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend
    static member Ones(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Ones(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend

    static member Create(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Create(value))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend

    static member Extend(a:Tensor, shape:int[]) =
        if a.Dim <> 0 then invalidArg "tensor" (sprintf "Expecting a 0d Tensor, received shape: %A" a.Shape)
        match a with
        | Tensor(ap) -> Tensor(ap.Extend(shape))
        | TensorF(ap,ad,at) ->
            let cp = Tensor.Extend(ap, shape)
            let cd = Tensor.Extend(ad, shape)
            TensorF(cp,cd,at)
        | TensorR(ap,_,_,_,at) ->
            let cp = Tensor.Extend(ap, shape)
            TensorR(cp, ref (a.Zero()), MakeTofT0(a), ref 0u, at)

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
        if a.ShapeEquals(b) then
            let inline fRaw(a,b) = a + b
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT(a,b)
            let inline dfTensorRevTC(a,b) = AddTTConst(a)
            let inline dfTensorRevCT(a,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a,b) = a + b
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Extend(ad, b.Shape)
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT0(b,a)
            let inline dfTensorRevTC(a,b) = AddTConstT0(a)
            let inline dfTensorRevCT(a,b) = AddTT0Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a,b) = a + b
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Extend(bd, a.Shape)
            let inline dfTensorRevTT(a,b) = AddTT0(a,b)
            let inline dfTensorRevTC(a,b) = AddTT0Const(a)
            let inline dfTensorRevCT(a,b) = AddTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape
    static member (+) (a:Tensor, b) = a + a.CreateWithSameType(b)
    static member (+) (a, b:Tensor) = b.CreateWithSameType(a) + b

    static member Sum (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.Sum()
        let inline fTensor(a) = Tensor.Sum(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Sum(ad)
        let inline dfTensorRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sum() = Tensor.Sum(t)

    member t.Reverse(?value:Tensor) =
        let value = defaultArg value (Tensor.OnesLike(t))
        t.ReverseReset()
        t.ReversePush(value)

    member inline t.Backward(value) = t.Reverse(value)

    member t.ReverseReset() =
        let rec reset (ts: Tensor list) =
            match ts with
            | [] -> ()
            | t :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    t.Derivative <- t.Zero()
                    t.Fanout <- t.Fanout + 1u
                    if t.Fanout = 1u then
                        match o with
                        | AddTT(a,b) -> reset (a::b::tt)
                        | AddTTConst(a) -> reset (a::tt)
                        | AddTT0(a,b) -> reset (a::b::tt)
                        | AddTT0Const(a) -> reset (a::tt)
                        | AddTConstT0(b) -> reset (b::tt)
                        | SumT(a) -> reset (a::tt)
                        | MakeTofT0(a) -> reset (a::tt)
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
                        match o with
                        | AddTT(a,b) -> push ((t.Derivative, a) :: (t.Derivative, b) :: tt)
                        | AddTTConst(a) -> push ((t.Derivative, a) :: tt)
                        | AddTT0(a,b) -> push ((t.Derivative, a) :: (t.Derivative.Sum(), b) :: tt)
                        | AddTT0Const(a) -> push ((t.Derivative, a) :: tt)
                        | AddTConstT0(b) -> push ((t.Derivative.Sum(), b) :: tt)
                        | SumT(a) -> push ((Tensor.Extend(t.Derivative, a.Shape), a) :: tt)
                        | MakeTofT0 (a) -> push ((t.Derivative.Sum(), a) :: tt)
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

    | SumT of Tensor
    | MakeTofT0 of Tensor
    | NewT

[<RequireQualifiedAccess>]
[<CompilationRepresentation (CompilationRepresentationFlags.ModuleSuffix)>]
module Tensor =
    let create (count:int) (value:float32) = Tensor.Create(Array.create count value)
    let zeroCreate (count:int) = Tensor.Create(Array.zeroCreate count)
    let init (count:int) (initializer:int->float32) = Tensor.Create(Array.init count initializer)
    let shape (t:Tensor) = t.Shape
    let dim (t:Tensor) = t.Dim