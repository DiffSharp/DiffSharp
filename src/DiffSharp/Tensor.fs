namespace DiffSharp

type Tensor =
    | Tensor of FloatTensor
    | TensorF of primal: Tensor * derivative: Tensor * tag: uint32
    | TensorR of primal: Tensor * derivative: (Tensor ref) * parentOp: TensorOp * fanOut: (uint32 ref) * tag: uint32
    
    member t.Primal =
        match t with
        | Tensor(_) -> t
        | TensorF(ap,_,_) -> ap
        | TensorR(ap,_,_,_,_) -> ap

    member t.PrimalDeep =
        let rec p x =
            match x with
            | Tensor(_) -> x
            | TensorF(xp,_,_) -> p xp
            | TensorR(xp,_,_,_,_) -> p xp
        p t

    member t.Derivative
        with get() =
            match t with
            | Tensor(_) -> failwith "Cannot get derivative of constant Tensor"
            | TensorF(_,ad,_) -> ad
            | TensorR(_,ad,_,_,_) -> !ad
        and set(value) =
            match t with
            | Tensor(_) -> failwith "Cannot set derivative of constant Tensor"
            | TensorF(_,ad,_) -> failwith "Cannot set derivative of TensorF"
            | TensorR(_,ad,_,_,_) -> ad := value

    member t.Shape = match t.PrimalDeep with Tensor(ap) -> ap.Shape
    static member Zero = Tensor(FloatTensor.From(0.f))
    static member Create(value:float32) = Tensor(FloatTensor.From(value))
    static member Create(values:float32[], shape:int64[]) = Tensor(FloatTensor.From(values, shape))

    static member inline OpUnary(a, ff, fd, df, r) =
        match a with
        | Tensor(ap)                 -> Tensor(ff(ap))
        | TensorF(ap, at, ai)        -> let cp = fd(ap) in TensorF(cp, df(cp, ap, at), ai)
        | TensorR(ap, _,  _,  _, ai) -> let cp = fd(ap) in TensorR(cp, ref (Tensor.Zero), r(a), ref 0u, ai)

    static member inline OpBinary(a, b, ff, fd, df_d_d, df_d_c, df_c_d, r_d_d, r_d_c, r_c_d) =
        match a, b with
        | Tensor(ap),                  Tensor(bp)                               -> Tensor(ff(ap, bp))
        | Tensor(_),                   TensorF(bp, bt, bi)                      -> let cp = fd(a, bp)  in TensorF(cp, df_c_d(cp, bp, bt), bi)
        | Tensor(_),                   TensorR(bp, _,  _,  _,  bi)              -> let cp = fd(a, bp)  in TensorR(cp, ref (Tensor.Zero), r_c_d(a, b), ref 0u, bi)
        | TensorF(ap, at, ai),         Tensor(_)                                -> let cp = fd(ap, b)  in TensorF(cp, df_d_c(cp, ap, at), ai)
        | TensorF(ap, at, ai),         TensorF(bp, bt, bi) when ai = bi         -> let cp = fd(ap, bp) in TensorF(cp, df_d_d(cp, ap, at, bp, bt), ai)
        | TensorF(ap, at, ai),         TensorF(_,  _,  bi) when ai > bi         -> let cp = fd(ap, b)  in TensorF(cp, df_d_c(cp, ap, at), ai)
        | TensorF(_,  _,  ai),         TensorF(bp, bt, bi) when ai < bi         -> let cp = fd(a, bp)  in TensorF(cp, df_c_d(cp, bp, bt), bi)
        | TensorF(_,  _,  ai),         TensorR(_,  _,  _,  _,  bi) when ai = bi -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorF(ap, at, ai),         TensorR(_,  _,  _,  _,  bi) when ai > bi -> let cp = fd(ap, b)  in TensorF(cp, df_d_c(cp, ap, at), ai)
        | TensorF(_,  _,  ai),         TensorR(bp, _,  _,  _,  bi) when ai < bi -> let cp = fd(a, bp)  in TensorR(cp, ref (Tensor.Zero), r_c_d(a, b), ref 0u, bi)
        | TensorR(ap, _,  _,  _,  ai), Tensor(_)                                -> let cp = fd(ap, b)  in TensorR(cp, ref (Tensor.Zero), r_d_c(a, b), ref 0u, ai)
        | TensorR(_,  _,  _,  _,  ai), TensorF(_,  _,  bi) when ai = bi         -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorR(ap, _,  _,  _,  ai), TensorF(_,  _,  bi) when ai > bi         -> let cp = fd(ap, b)  in TensorR(cp, ref (Tensor.Zero), r_d_c(a, b), ref 0u, ai)
        | TensorR(_,  _,  _,  _,  ai), TensorF(bp, bt, bi) when ai < bi         -> let cp = fd(a, bp)  in TensorF(cp, df_c_d(cp, bp, bt), bi)
        | TensorR(ap, _,  _,  _,  ai), TensorR(bp, _,  _,  _,  bi) when ai = bi -> let cp = fd(ap, bp) in TensorR(cp, ref (Tensor.Zero), r_d_d(a, b), ref 0u, ai)
        | TensorR(ap, _,  _,  _,  ai), TensorR(_,  _,  _,  _,  bi) when ai > bi -> let cp = fd(ap, b)  in TensorR(cp, ref (Tensor.Zero), r_d_c(a, b), ref 0u, ai)
        | TensorR(_,  _,  _,  _,  ai), TensorR(bp, _,  _,  _,  bi) when ai < bi -> let cp = fd(a, bp)  in TensorR(cp, ref (Tensor.Zero), r_c_d(a, b), ref 0u, bi)

    static member (+) (a:Tensor, b:Tensor) =
        let inline ff(a, b) = a + b
        let inline fd(a, b) = a + b
        let inline df_d_d(cp, ap, at, bp, bt) = at + bt
        let inline df_d_c(cp, ap, at) = at
        let inline df_c_d(cp, bp, bt) = bt
        let inline r_d_d(a, b) = Add_T_T(a, b)
        let inline r_d_c(a, b) = Add_T_TCons(a)
        let inline r_c_d(a, b) = Add_T_TCons(b)
        Tensor.OpBinary(a, b, ff, fd, df_d_d, df_d_c, df_c_d, r_d_d, r_d_c, r_c_d)

and TensorOp =
    | Add_T_T of Tensor * Tensor
    | Add_T_TCons of Tensor

