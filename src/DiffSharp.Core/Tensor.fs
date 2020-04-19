namespace DiffSharp
open DiffSharp.Backend
open DiffSharp.Util
open System.IO
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary

#nowarn "1182" // turn off compiler-generated unused variable warnings in this file only

[<CustomEquality; CustomComparison>]
type Tensor = 
    | Tensor of primalRaw:RawTensor
    | TensorF of primal:Tensor * derivative:Tensor * nestingTag:uint32
    | TensorR of primal:Tensor * derivative:(Tensor ref) * parentOperation:TensorOp * fanout:(uint32 ref) * nestingTag:uint32

    member t.primal =
        match t with
        | Tensor(_) -> t
        | TensorF(tp,_,_) -> tp
        | TensorR(tp,_,_,_,_) -> tp

    member t.primalDeep =
        match t with
        | Tensor(_) -> t
        | TensorF(tp,_,_) -> tp.primalDeep
        | TensorR(tp,_,_,_,_) -> tp.primalDeep

    member t.primalRaw =
        match t with
        | Tensor(tp) -> tp
        | TensorF(tp,_,_) -> tp.primalRaw
        | TensorR(tp,_,_,_,_) -> tp.primalRaw

    member t.depth =
        let rec depth x d =
            match x with
            | Tensor(_) -> d
            | TensorF(tp,_,_) -> depth tp (d + 1)
            | TensorR(tp,_,_,_,_) -> depth tp (d + 1)
        depth t 0

    member t.derivative
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

    member t.derivativeDeep =
        match t with
        | Tensor(_) -> failwith "Cannot get derivative of constant Tensor"
        | TensorF(_,td,_) -> 
            match td with
            | Tensor(_) -> td
            | _ -> td.derivativeDeep
        | TensorR(_,td,_,_,_) -> 
            match !td with
            | Tensor(_) -> !td
            | _ -> (!td).derivativeDeep

    member t.fanout
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

    member t.forwardDiff(derivative:Tensor, ?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        if t.shape = derivative.shape then TensorF(t, derivative, tag) else failwithf "Expecting derivative of same shape with primal. primal: %A, derivative: %A" t derivative
    member t.reverseDiff(?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        TensorR(t, ref (t.zeroLike()), NewT, ref 0u, tag)
    member t.noDiff() = Tensor(t.primalRaw)
    member t.shape = t.primalRaw.Shape
    member t.dim = t.primalRaw.Dim
    member t.nelement = t.primalRaw.Nelement
    member t.toArray() = t.primalRaw.ToArray()
    member t.toScalar() = t.primalRaw.ToValue()
    member t1.isSameDiffType(t2:Tensor) =
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
    member t.save(fileName:string) =
        let formatter = BinaryFormatter()
        let fs = new FileStream(fileName, FileMode.Create)
        try
            formatter.Serialize(fs, t)
        with
        | :? SerializationException as e -> failwithf "Cannot save Tensor. %A" e.Message
        fs.Close()
    static member load(fileName:string) =
        let formatter = BinaryFormatter()
        let fs = new FileStream(fileName, FileMode.Open)
        try
            let t = formatter.Deserialize(fs) :?> Tensor
            fs.Close()
            t
        with
        | :? SerializationException as e -> failwithf "Cannot load Tensor. %A" e.Message


    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.primalRaw.Equals(tensor.primalRaw)
        | _ -> false
    member t.allclose(tensor:Tensor, ?relativeTolerance, ?absoluteTolerance) =
        let relativeTolerance = defaultArg relativeTolerance 1e-5
        let absoluteTolerance = defaultArg absoluteTolerance 1e-8
        t.primalRaw.AllClose(tensor.primalRaw, relativeTolerance, absoluteTolerance)
    override t.GetHashCode() = hash t.primalRaw
    interface System.IComparable with
        override t.CompareTo(other) =
            match other with
            | :? Tensor as tensor -> 
                if t.dim = tensor.dim && t.dim = 0 then
                    t.primalRaw.CompareTo(tensor.primalRaw)
                else
                    failwith "Cannot compare non-scalar Tensors"
            | _ -> failwith "Cannot compare Tensor with another type"
    static member op_Explicit(tensor:Tensor):'a = downcast tensor.toScalar()

    member a.zerosLike(?shape:seq<int>) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.Zeros(shape |> Array.ofSeq))
    member a.onesLike(?shape:seq<int>) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.Ones(shape |> Array.ofSeq))
    member a.fullLike(shape:seq<int>, value:obj) = 
        let value = if value :? Tensor then (value:?>Tensor).toScalar() else value    
        Tensor(a.primalRaw.Full(shape |> Array.ofSeq, value))
    member a.randLike(?shape:seq<int>) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.Random(shape |> Array.ofSeq))
    member a.randnLike(?shape:seq<int>) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.RandomNormal(shape |> Array.ofSeq))
    member a.zeroLike() = Tensor(a.primalRaw.Zero())
    member a.oneLike() = Tensor(a.primalRaw.One())
    member a.arangeLike(endVal:float, ?startVal:float, ?step:float) =
        let startVal = defaultArg startVal 0.
        let step = defaultArg step 1.
        let length = (endVal - startVal) / step |> ceil |> int
        let v = Array.init length (fun i -> startVal + float(i) * step)
        a.like(box v)
    member a.like(value) = Tensor(a.primalRaw.Create(value))
    member a.clone() = Tensor(a.primalRaw.Clone())
    member a.onehotLike(length:int, hot:int) =
        if hot < 0 || hot >= length then failwithf "Expecting 0 <= hot < length"
        a.zerosLike([|length|]).addSlice([|hot|], a.onesLike([|1|]))
    member a.lt(b:Tensor) = Tensor(a.primalRaw.LtTT(b.primalRaw))
    member a.gt(b:Tensor) = Tensor(a.primalRaw.GtTT(b.primalRaw))
    member a.le(b:Tensor) =Tensor(a.primalRaw.LeTT(b.primalRaw))
    member a.ge(b:Tensor) = Tensor(a.primalRaw.GeTT(b.primalRaw))
    member a.maxIndex() = a.primalRaw.MaxIndexT()
    member a.minIndex() = a.primalRaw.MinIndexT()
    member a.max() = a.[a.maxIndex()]
    member a.min() = a.[a.minIndex()]
    member a.max(b:Tensor) = ((a + b) + Tensor.Abs(b - a)) / 2.
    member a.min(b:Tensor) = ((a + b) - Tensor.Abs(a - b)) / 2.

    member a.diagonal(?offset:int, ?dim1:int, ?dim2:int) =
        if a.dim < 2 then failwithf "Tensor must be at least 2-dimensional"
        let offset = defaultArg offset 0
        let dim1 = defaultArg dim1 0
        let dim2 = defaultArg dim2 1
        let mutable finished = false
        let mutable d = []
        let mutable i = 0
        let mutable j = offset
        while not finished do
            if i >= a.shape.[dim1] || j >= a.shape.[dim2] then 
                finished <- true
            elif j >= 0 then
                // let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
                let bounds = Array2D.init (a.dim) 3 (fun ii jj -> 
                                                        if ii = dim1 then
                                                            if jj < 2 then i else 1
                                                        elif ii = dim2 then
                                                            if jj < 2 then j else 1
                                                        else
                                                            if jj = 0 then 0
                                                            elif jj = 1 then a.shape.[ii]-1
                                                            else 0
                                                        )
                d <- [a.GetSlice(bounds)] |> List.append d
            i <- i + 1
            j <- j + 1
        if d |> List.isEmpty then failwithf "Empty diagonal"
        Tensor.stack(d)

    member a.trace() = let d:Tensor = a.diagonal() in d.sum()

    member a.expand(newShape:seq<int>) =
        let newShape = newShape|>Seq.toArray
        if a.shape = newShape then a else
        match a with
        | Tensor(ap) -> Tensor(ap.Expand(newShape))
        | TensorF(ap,ad,at) ->
            let cp = ap.expand(newShape)
            let cd = ad.expand(newShape)
            TensorF(cp,cd,at)
        | TensorR(ap,_,_,_,at) ->
            let cp = ap.expand(newShape)
            TensorR(cp, ref (a.zeroLike()), ExpandT(a), ref 0u, at)

    member internal t.GetSlice(bounds:int[,]) =
        // printfn "t.GetSlice bounds\n %A" bounds
        if t.dim = 0 then failwith "Cannot slice a scalar Tensor"
        let fullBounds = Array2D.init t.dim 3 (fun i j -> if j=0 then 0 elif j=1 then t.shape.[i]-1 else 0)
        bounds |> Array2D.iteri (fun i j v -> 
            if j=1 && v >= t.shape.[i] then failwithf "Index outside the bounds of Tensor shape %A" t.shape
            fullBounds.[i, j] <- v)
        // printfn "t.GetSlice fullBounds\n %A" fullBounds
        match t with
        | Tensor(ap) -> Tensor(ap.GetSlice(fullBounds))
        | TensorF(ap,ad,at) -> TensorF(ap.GetSlice(fullBounds), ad.GetSlice(fullBounds), at)
        | TensorR(ap,_,_,_,at) -> TensorR(ap.GetSlice(fullBounds), ref (ap.zeroLike()), SliceT(t, fullBounds), ref 0u, at)

    member t.Item
        with get([<System.ParamArray>] index:int[]) =
            if t.dim = 0 then failwith "Cannot index a scalar Tensor"
            if index.Length > t.dim then failwithf "Expecting an index with <=%i dimensions" t.dim
            let bounds = Array2D.init index.Length 3 (fun i j -> if j=2 then 1 else index.[i])
            t.GetSlice(bounds)

    static member create(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) =
        let array, shape = value |> flatArrayAndShape<Tensor> // support creation of new Tensor from a structure holding scalar Tensors
        if notNull array then 
            let array = array |> Array.map float32
            let value = arrayND shape (fun ii -> array.[indexToFlatIndex shape ii])
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))
        else
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))        

    static member stack(tensors:seq<Tensor>, ?dim:int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        if tensors.Length = 0 then failwithf "Expecting a non-empty sequence of Tensors"
        // TODO: check if all Tensors are of the same type (Tensor, TensorF, or TensorR) and have the same nesting tag
        let shapes = tensors |> Seq.map (fun t -> t.shape)
        checkCanStack shapes
        match Seq.head tensors with
        | Tensor(ap) -> Tensor(ap.StackTs((tensors |> Array.map (fun t -> t.primalRaw)), dim))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let ad = tensors |> Seq.map (fun t -> t.derivative)
            TensorF(Tensor.stack(ap,dim=dim), Tensor.stack(ad,dim=dim), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let cp = Tensor.stack(ap,dim=dim)
            TensorR(cp, ref (cp.zeroLike()), StackTs(tensors, dim), ref 0u, at)

    member a.unstack (?dim:int) =
        let dim = defaultArg dim 0 
        checkCanUnstack a.dim
        match a with
        | Tensor(ap) -> ap.UnstackT(dim) |> Array.map Tensor
        | TensorF(ap,ad,at) -> Array.map2 (fun p d -> TensorF(p,d,at)) (ap.unstack(dim)) (ad.unstack(dim))
        | TensorR(ap,_,_,_,at) -> Array.mapi (fun i p -> TensorR(p, ref (p.zeroLike()), UnstackT(a, dim, i), ref 0u, at)) (ap.unstack(dim))

    static member cat(tensors:seq<Tensor>, ?dim: int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        // TODO: check if all Tensors are of the same type (Tensor, TensorF, or TensorR) and have the same nesting tag
        match Seq.head tensors with
        | Tensor(ap) -> Tensor(ap .CatTs((tensors |> Array.map (fun t -> t.primalRaw)), dim))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let ad = tensors |> Seq.map (fun t -> t.derivative)
            TensorF(Tensor.cat(ap, dim=dim), Tensor.cat(ad, dim=dim), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let cp = Tensor.cat(ap, dim=dim)
            TensorR(cp, ref (cp.zeroLike()), CatTs(tensors, dim), ref 0u, at)

    member a.split (sizes: seq<int>, ?dim: int) =
        let dim = defaultArg dim 0
        let sizes = sizes |> Seq.toArray
        match a with
        | Tensor(ap) -> ap.SplitT(sizes, dim=dim) |> Array.map Tensor
        | TensorF(ap,ad,at) -> Array.map2 (fun p d -> TensorF(p,d,at)) (ap.split(sizes)) (ad.split(sizes, dim=dim))
        | TensorR(ap,_,_,_,at) -> Array.mapi (fun i p -> TensorR(p, ref (p.zeroLike()), SplitT(a, sizes, dim, i), ref 0u, at)) (ap.split(sizes, dim=dim))

    static member inline OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev) =
        match a with
        | Tensor(ap)           -> Tensor(fRaw(ap))
        | TensorF(ap,ad,at)    -> let cp = fTensor(ap) in TensorF(cp, dfTensorFwd(cp,ap,ad), at)
        | TensorR(ap,_,_,_,at) -> let cp = fTensor(ap) in TensorR(cp, ref (a.zeroLike()), dfTensorRev(a), ref 0u, at)

    static member inline OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT) =
        match a, b with
        | Tensor(ap),           Tensor(bp)                      -> Tensor(fRaw(ap, bp))
        | Tensor(_),            TensorF(bp,bd,bt)               -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | Tensor(_),            TensorR(bp,_,_,_,bt)            -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.zeroLike()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorF(ap,ad,at),    Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(ap,ad,at),    TensorF(bp,bd,bt)    when at=bt -> let cp = fTensor(ap,bp) in TensorF(cp, dfTensorFwdTT(cp,ap,ad,bp,bd), at)
        | TensorF(ap,ad,at),    TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | TensorF(_,_,at),      TensorR(_,_,_,_,bt)  when at=bt -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorF(ap,ad,at),    TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.zeroLike()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorR(ap,_,_,_,at), Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.zeroLike()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(_,_,bt)      when at=bt -> failwith "Cannot have TensorR and TensorF in the same nesting level"
        | TensorR(ap,_,_,_,at), TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap, b) in TensorR(cp, ref (a.zeroLike()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp, bp, bd), bt)
        | TensorR(ap,_,_,_,at), TensorR(bp,_,_,_,bt) when at=bt -> let cp = fTensor(ap,bp) in TensorR(cp, ref (a.zeroLike()), dfTensorRevTT(a,b), ref 0u, at)
        | TensorR(ap,_,_,_,at), TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.zeroLike()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.zeroLike()), dfTensorRevCT(a,b), ref 0u, bt)
        | _ -> failwith "Unexpected combination of Tensors" // Won't happen, added for suppressing "incomplete matches" warning

    static member (+) (a:Tensor, b:Tensor) =
        if a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.AddTT(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT(a,b)
            let inline dfTensorRevTC(a,b) = AddTTConst(a)
            let inline dfTensorRevCT(a,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.AddTT0(a)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad:Tensor) = ad.expand(b.shape)
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT0(b,a)
            let inline dfTensorRevTC(a,b) = AddTConstT0(a)
            let inline dfTensorRevCT(a,b) = AddTT0Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.AddTT0(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd:Tensor) = bd.expand(a.shape)
            let inline dfTensorRevTT(a,b) = AddTT0(a,b)
            let inline dfTensorRevTC(a,b) = AddTT0Const(a)
            let inline dfTensorRevCT(a,b) = AddTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 2 && b.dim = 1 && a.shape.[1] = b.shape.[0] then
            let inline fRaw(a:RawTensor,b) = a.AddT2T1(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp:Tensor,bp,bd) = cp.zerosLike() + bd
            let inline dfTensorRevTT(a,b) = AddT2T1(a,b)
            let inline dfTensorRevTC(a,b) = AddT2T1Const(a)
            let inline dfTensorRevCT(a,b) = AddT2ConstT1(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 1 && b.dim = 2 && a.shape.[0] = b.shape.[1] then
            let inline fRaw(a,b:RawTensor) = b.AddT2T1(a)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp:Tensor,ap,ad) = ad + cp.zerosLike()
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddT2T1(b,a)
            let inline dfTensorRevTC(a,b) = AddT2ConstT1(a)
            let inline dfTensorRevCT(a,b) = AddT2T1Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = broadcastShapes2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded + bExpanded
    static member (+) (a:Tensor, b) = a + a.like(b)
    static member (+) (a, b:Tensor) = b.like(a) + b
    member a.add(b:Tensor) = a + b
    member a.add(b) = a + a.like(b)

    static member (-) (a:Tensor, b:Tensor) =
        if a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.SubTT(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubTT(a,b)
            let inline dfTensorRevTC(a,b) = SubTTConst(a)
            let inline dfTensorRevCT(a,b) = SubTConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubT0T(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad:Tensor) = ad.expand(b.shape)
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubT0T(a,b)
            let inline dfTensorRevTC(a,b) = SubT0TConst(a)
            let inline dfTensorRevCT(a,b) = SubT0ConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubTT0(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd:Tensor) = (-bd).expand(a.shape)
            let inline dfTensorRevTT(a,b) = SubTT0(a,b)
            let inline dfTensorRevTC(a,b) = SubTT0Const(a)
            let inline dfTensorRevCT(a,b) = SubTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = broadcastShapes2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded - bExpanded
    static member (-) (a:Tensor, b) = a - a.like(b)
    static member (-) (a, b:Tensor) = b.like(a) - b
    member a.sub(b:Tensor) = a - b
    member a.sub(b) = a - a.like(b)

    static member (*) (a:Tensor, b:Tensor) =
        if a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.MulTT(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT(a,b)
            let inline dfTensorRevTC(a,b) = MulTTConst(a,b)
            let inline dfTensorRevCT(a,b) = MulTTConst(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.MulTT0(a)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(b,a)
            let inline dfTensorRevTC(a,b) = MulTConstT0(a,b)
            let inline dfTensorRevCT(a,b) = MulTT0Const(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.MulTT0(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(a,b)
            let inline dfTensorRevTC(a,b) = MulTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = MulTConstT0(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = broadcastShapes2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded * bExpanded
    static member (*) (a:Tensor, b) = a * a.like(b)
    static member (*) (a, b:Tensor) = b.like(a) * b
    member a.mul(b:Tensor) = a * b
    member a.mul(b) = a * a.like(b)

    static member (/) (a:Tensor, b:Tensor) =
        if a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.DivTT(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT(a,b)
            let inline dfTensorRevTC(a,b) = DivTTConst(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivT0T(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivT0T(a,b)
            let inline dfTensorRevTC(a,b) = DivT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = DivT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivTT0(b)
            let inline fTensor(a:Tensor,b:Tensor) = a / b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let inline dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT0(a,b)
            let inline dfTensorRevTC(a,b) = DivTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = broadcastShapes2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded / bExpanded
    static member (/) (a:Tensor, b) = a / a.like(b)
    static member (/) (a, b:Tensor) = b.like(a) / b
    member a.div(b:Tensor) = a / b
    member a.div(b) = a / a.like(b)

    static member Pow (a:Tensor, b:Tensor) =
        if a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.PowTT(b)
            let inline fTensor(a:Tensor,b:Tensor) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT(a,b)
            let inline dfTensorRevTC(a,b) = PowTTConst(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowT0T(b)
            let inline fTensor(a:Tensor,b:Tensor) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowT0T(a,b)
            let inline dfTensorRevTC(a,b) = PowT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = PowT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowTT0(b)
            let inline fTensor(a:Tensor,b:Tensor) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT0(a,b)
            let inline dfTensorRevTC(a,b) = PowTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = broadcastShapes2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            Tensor.Pow(aExpanded, bExpanded)

    static member Pow (a:Tensor, b:float) = a ** a.like(b)
    static member Pow (a:Tensor, b:int) = a ** a.like(b)
    static member Pow (a:Tensor, b) = a ** a.like(b)
    static member Pow (a:float, b:Tensor) = b.like(a) ** b
    static member Pow (a:int, b:Tensor) = b.like(a) ** b
    static member Pow (a, b:Tensor) = b.like(a) ** b
    member a.pow(b:Tensor) = a ** b
    member a.pow(b) = a ** a.like(b)

    member a.matmul (b:Tensor) =
        checkCanMatmul a.shape b.shape
        let inline fRaw(a:RawTensor,b) = a.MatMulT2T2(b)
        let inline fTensor(a:Tensor,b) = a.matmul(b)
        let inline dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp,bd) = ad.matmul(bp) + ap.matmul(bd)
        let inline dfTensorFwdTC(cp,ap,ad:Tensor) = ad.matmul(b)
        let inline dfTensorFwdCT(cp,bp,bd) = a.matmul(bd)
        let inline dfTensorRevTT(a,b) = MatMulT2T2(a,b)
        let inline dfTensorRevTC(a,b) = MatMulT2T2Const(a,b)
        let inline dfTensorRevCT(a,b) = MatMulT2ConstT2(a,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    member a.dot(b:Tensor) =
        checkCanDot a.shape b.shape
        let a:Tensor = a.view([1;a.nelement])
        let b:Tensor = b.view([b.nelement;1])
        a.matmul(b).view([])

    static member (~-) (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.NegT()
        let inline fTensor(a) = -a
        let inline dfTensorFwd(cp,ap,ad) = -ad
        let inline dfTensorRev(a) = NegT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member a.neg() = -a

    member a.sum() =
        let inline fRaw(a:RawTensor) = a.SumT()
        let inline fTensor(a:Tensor) = a.sum()
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.sum()
        let inline dfTensorRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    // TODO: this can be implemented in a more memory efficient way by pushing the sum operation to the RawTensor level and implementing the derivatives using general broadcasting when it's available
    member a.sum(dim:int, ?keepDim:bool) =
       let keepDim = defaultArg keepDim false
       let res =
        if dim = 0 && a.dim = 0 then a
        else
            if dim >= a.dim || dim < 0 then failwithf "Expecting dim to be between 0 and %A" a.dim
            let sBounds = Array2D.init a.dim 3 (fun i j -> if j=0 then 0 elif j=1 then a.shape.[i]-1 else 0)
            sBounds.[dim, 1] <- 0
            sBounds.[dim, 2] <- 1
            let mutable s = a.zerosLike().GetSlice(sBounds)
            for i=0 to a.shape.[dim]-1 do
                sBounds.[dim,0] <- i
                sBounds.[dim,1] <- i
                sBounds.[dim,2] <- 1
                s <- s + a.GetSlice(sBounds)
            s
       if keepDim then res.unsqueeze(dim) else res

    member a.sum(dim, ?keepDim) = a.sum(dim, ?keepDim=keepDim)

    /// Reduce the dimensionality via summation until we reach `newShape`.  An expansion
    /// from newShape to shape must be possible.
    member a.sumToSize(newShape:int[]) =
        let oldShape = a.shape
        if oldShape = newShape then a
        elif newShape.Length = 0 then a.sum()
        else
            checkCanExpandShape newShape oldShape
            let trim = oldShape.Length - newShape.Length
            let mutable result = a
            // collapse the eliminated dimensions
            for _dim in 0 .. trim-1 do 
                result <- result.sum(0, keepDim=false)
            // reduce the squeezed dimensions
            for dim in 0 .. newShape.Length-1 do 
                if oldShape.[trim+dim] <> newShape.[dim] then 
                    result <- result.sum(dim, keepDim=true)
            result

    member a.mean() = a.sum() / a.nelement

    member a.mean(dim:int, ?keepDim:bool) = 
        if dim = 0 && a.dim = 0 then a
        else a.sum(dim, ?keepDim=keepDim) / a.shape.[dim]

    // This is the two-pass algorithm better than the naive algorithm
    member a.variance() = let a' = a - a.mean() in (a' * a').sum() / (a.nelement - 1)

    // TODO: this is the naive algorithm, can be improved for better numerical stability
    member a.variance(dim:int, ?keepDim:bool) =
        let keepDim = defaultArg keepDim false
        if dim >= a.dim || dim < 0 then failwithf "Expecting dim to be between 0 and %A" a.dim
        let sBounds = Array2D.init a.dim 3 (fun i j -> if j=0 then 0 elif j=1 then a.shape.[i]-1 else 0)
        sBounds.[dim, 1] <- 0
        sBounds.[dim, 2] <- 1
        let mutable s = a.zerosLike().GetSlice(sBounds)
        let mutable sSquare = a.zerosLike().GetSlice(sBounds)
        let n = a.shape.[dim]
        for i=0 to n-1 do
            sBounds.[dim,0] <- i
            sBounds.[dim,1] <- i
            sBounds.[dim,2] <- 1
            let slice = a.GetSlice(sBounds)
            s <- s + slice
            sSquare <- sSquare + slice * slice
        let res = (sSquare - (s * s) / n) / (n - 1)
        if keepDim then res.unsqueeze(dim) else res

    member a.stddev(dim:int, ?keepDim) = a.variance(dim, ?keepDim=keepDim) |> Tensor.Sqrt

    member a.stddev() = a.variance() |> Tensor.Sqrt

    // This is useful to keep as a special case of sum for performance reasons because it's involved in reverse mode of broadcasting addition of bias in NN linear layers
    member internal a.sumT2Dim0() =
        let inline fRaw(a:RawTensor) = a.SumT2Dim0()
        let inline fTensor(a:Tensor) = a.sumT2Dim0()
        let inline dfTensorFwd(cp,ap,ad:Tensor):Tensor = ad.sumT2Dim0()
        let inline dfTensorRev(a) = SumT2Dim0(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    
    member a.transpose() =
        checkCanTranspose a.dim
        let inline fRaw(a:RawTensor) = a.TransposeT2()
        let inline fTensor(a:Tensor) = a.transpose()
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.transpose()
        let inline dfTensorRev(a) = TransposeT2(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.squeeze(?dim:int) =
        let dim = defaultArg dim -1
        let inline fRaw(a:RawTensor) = a.SqueezeT(dim)
        let inline fTensor(a:Tensor) = a.squeeze(dim)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.squeeze(dim)
        let inline dfTensorRev(a) = SqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.unsqueeze(dim:int) =
        let inline fRaw(a:RawTensor) = a.UnsqueezeT(dim)
        let inline fTensor(a:Tensor) = a.unsqueeze(dim)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.unsqueeze(dim)
        let inline dfTensorRev(a) = UnsqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.flip(dims:seq<int>) =
        let dims = dims |> Array.ofSeq
        checkCanFlip a.dim dims
        let inline fRaw(a:RawTensor) = a.FlipT(dims)
        let inline fTensor(a:Tensor) = a.flip(dims)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.flip(dims)
        let inline dfTensorRev(a) = FlipT(a, dims)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.dilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        checkCanDilate a.dim dilations
        let inline fRaw(a:RawTensor) = a.DilateT(dilations)
        let inline fTensor(a:Tensor) = a.dilate(dilations)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.dilate(dilations)
        let inline dfTensorRev(a) = DilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.undilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        let inline fRaw(a:RawTensor) = a.UndilateT(dilations)
        let inline fTensor(a:Tensor) = a.undilate(dilations)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.undilate(dilations)
        let inline dfTensorRev(a) = UndilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.repeat(dim:int, times:int) =
        if a.shape.[dim] <> 1 then failwithf "Expecting Tensor's shape at dim to be 1, received Tensor with shape %A and dim %A" a.shape dim
        let newShape = a.shape |> Array.copy
        newShape.[dim] <- times
        let mutable ret = a.zerosLike(newShape)
        let location = Array.create a.dim 0
        for i=0 to times-1 do
            location.[dim] <- i
            ret <- ret.addSlice(location, a)
        ret

    member a.view(shape:seq<int>) =
        let shape = shape |> Seq.toArray |> shapeComplete a.nelement  // Handles -1 semantics
        checkCanView a.shape shape
        let inline fRaw(a:RawTensor) = a.ViewT(shape)
        let inline fTensor(a:Tensor) = a.view(shape)
        let inline dfTensorFwd(cp,ap,ad:Tensor) = ad.view(shape)
        let inline dfTensorRev(a) = ViewT(a, a.shape)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.view(shape:int) = t.view([|shape|])

    member a.viewAs(b:Tensor) = a.view(b.shape)

    member a.sign() =
        let inline fRaw(a:RawTensor) = a.SignT()
        let inline fTensor(a:Tensor) = a.sign()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let inline dfTensorRev(a) = SignT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    // static member Sign(a:Tensor) = a.sign() // not supported becaose FSharp.Core sign operator returns int

    member a.floor() =
        let inline fRaw(a:RawTensor) = a.FloorT()
        let inline fTensor(a:Tensor) = a.floor()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let inline dfTensorRev(a) = FloorT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Floor(a:Tensor) = a.floor() // needed for FSharp.Core floor operator overload

    member a.ceil() =
        let inline fRaw(a:RawTensor) = a.CeilT()
        let inline fTensor(a:Tensor) = a.ceil()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let inline dfTensorRev(a) = CeilT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Ceiling(a:Tensor) = a.ceil() // needed for FSharp.Core ceil operator overload

    member a.round() =
        let inline fRaw(a:RawTensor) = a.RoundT()
        let inline fTensor(a:Tensor) = a.round()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let inline dfTensorRev(a) = RoundT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Round(a:Tensor) = a.round() // needed for FSharp.Core round operator overload

    member a.abs() =
        let inline fRaw(a:RawTensor) = a.AbsT()
        let inline fTensor(a:Tensor) = a.abs()
        let inline dfTensorFwd(cp,ap:Tensor,ad) = ad * ap.sign()
        let inline dfTensorRev(a) = AbsT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Abs(a:Tensor) = a.abs() // needed for FSharp.Core abs operator overload

    member a.relu() =
        let inline fRaw(a:RawTensor) = a.ReluT()
        let inline fTensor(a:Tensor) = a.relu()
        let inline dfTensorFwd(cp,ap:Tensor,ad) = let sap = ap.sign() in ad * sap.abs() * (1. + sap) / 2.
        let inline dfTensorRev(a) = ReluT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.leakyRelu(?negativeSlope:float) =
        let negativeSlope = defaultArg negativeSlope 0.01
        let zeros = a.zerosLike() in zeros.max(a) + negativeSlope * zeros.min(a)

    member a.sigmoid() =
        let inline fRaw(a:RawTensor) = a.SigmoidT()
        let inline fTensor(a:Tensor) = a.sigmoid()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad * cp * (1. - cp)
        let inline dfTensorRev(a) = SigmoidT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.exp() =
        let inline fRaw(a:RawTensor) = a.ExpT()
        let inline fTensor(a:Tensor) = a.exp()
        let inline dfTensorFwd(cp,ap,ad) = ad * cp
        let inline dfTensorRev(a) = ExpT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Exp(a:Tensor) = a.exp() // needed for FSharp.Core exp operator overload

    member a.log() =
        let inline fRaw(a:RawTensor) = a.LogT()
        let inline fTensor(a:Tensor) = a.log()
        let inline dfTensorFwd(cp,ap,ad) = ad / ap
        let inline dfTensorRev(a) = LogT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Log(a:Tensor) = a.log() // needed for FSharp.Core log operator overload

    member a.log10() =
        let inline fRaw(a:RawTensor) = a.Log10T()
        let inline fTensor(a:Tensor) = a.log10()
        let inline dfTensorFwd(cp,ap:Tensor,ad) = ad / (ap * log10Val)
        let inline dfTensorRev(a) = Log10T(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Log10(a:Tensor) = a.log10() // needed for FSharp.Core log10 operator overload

    member a.sqrt() =
        let inline fRaw(a:RawTensor) = a.SqrtT()
        let inline fTensor(a:Tensor) = a.sqrt()
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad / (2. * cp)
        let inline dfTensorRev(a) = SqrtT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sqrt(a:Tensor) = a.sqrt() // needed for FSharp.Core sqrt operator overload

    member a.sin() =
        let inline fRaw(a:RawTensor) = a.SinT()
        let inline fTensor(a:Tensor) = a.sin()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.cos()
        let inline dfTensorRev(a) = SinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sin(a:Tensor) = a.sin() // needed for FSharp.Core sin operator overload

    member a.cos() =
        let inline fRaw(a:RawTensor) = a.CosT()
        let inline fTensor(a:Tensor) = a.cos()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = -ad * ap.sin()
        let inline dfTensorRev(a) = CosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Cos(a:Tensor) = a.cos() // needed for FSharp.Core cos operator overload

    member a.tan() =
        let inline fRaw(a:RawTensor) = a.TanT()
        let inline fTensor(a:Tensor) = a.tan()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = let cosap = ap.cos() in ad / (cosap * cosap)
        let inline dfTensorRev(a) = TanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Tan(a:Tensor) = a.tan() // needed for FSharp.Core tan operator overload

    member a.sinh() =
        let inline fRaw(a:RawTensor) = a.SinhT()
        let inline fTensor(a:Tensor) = a.sinh()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.cosh()
        let inline dfTensorRev(a) = SinhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sinh(a:Tensor) = a.sinh() // needed for FSharp.Core sinh operator overload

    member a.cosh() =
        let inline fRaw(a:RawTensor) = a.CoshT()
        let inline fTensor(a:Tensor) = a.cosh()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.sinh()
        let inline dfTensorRev(a) = CoshT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Cosh(a:Tensor) = a.cosh() // needed for FSharp.Core cosh operator overload

    member a.tanh() =
        let inline fRaw(a:RawTensor) = a.TanhT()
        let inline fTensor(a:Tensor) = a.tanh()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = let coshap = ap.cosh() in ad / (coshap * coshap)
        let inline dfTensorRev(a) = TanhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Tanh(a:Tensor) = a.tanh() // needed for FSharp.Core tanh operator overload

    member a.asin() =
        let inline fRaw(a:RawTensor) = a.AsinT()
        let inline fTensor(a:Tensor) = a.asin()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / (1. - ap*ap).sqrt()
        let inline dfTensorRev(a) = AsinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Asin(a:Tensor) = a.asin() // needed for FSharp.Core asin operator overload

    member a.acos() =
        let inline fRaw(a:RawTensor) = a.AcosT()
        let inline fTensor(a:Tensor) = a.acos()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = -ad / (1. - ap*ap).sqrt()
        let inline dfTensorRev(a) = AcosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Acos(a:Tensor) = a.acos() // needed for FSharp.Core acos operator overload

    member a.atan() =
        let inline fRaw(a:RawTensor) = a.AtanT()
        let inline fTensor(a:Tensor) = a.atan()
        let inline dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / (1. + ap*ap)
        let inline dfTensorRev(a) = AtanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Atan(a:Tensor) = a.atan() // needed for FSharp.Core atan operator overload

    member a.addSlice(location:seq<int>, b:Tensor) =
        let location = location |> Seq.toArray
        checkCanAddSlice a.shape location b.shape
        let inline fRaw(a:RawTensor,b) = a.AddTTSlice(location, b)
        let inline fTensor(a:Tensor,b) = a.addSlice(location, b)
        let inline dfTensorFwdTT(cp,ap,ad:Tensor,bp,bd) = ad.addSlice(location, bd)
        let inline dfTensorFwdTC(cp,ap,ad) = ad
        let inline dfTensorFwdCT(cp:Tensor,bp,bd) = cp.zerosLike().addSlice(location, bd)
        let inline dfTensorRevTT(a,b) = AddTTSlice(a,location,b)
        let inline dfTensorRevTC(a,b) = AddTTConstSlice(a)
        let inline dfTensorRevCT(a,b) = AddTConstTSlice(location,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    member a.softmax(dim:int) =
        if dim < 0 || dim >= a.dim then failwithf "Expecting 0 <= dim < a.dim, received %A, %A" dim a.dim
        let e = (a - a.max().noDiff()).exp()
        let esum = e.sum(dim, keepDim=true).repeat(dim, a.shape.[dim])
        e / esum

    member a.logsumexp(dim:int, ?keepDim:bool) =
        let keepDim = defaultArg keepDim false
        let amax = a.max().noDiff()
        let e = (a - amax).exp()
        let res = amax + e.sum(dim).log()
        if keepDim then res.unsqueeze(dim) else res

    member a.mseLoss(b:Tensor) = let z = a - b in (z * z).mean()

    member a.conv1d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int) =
        // a: input, b: filter
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let dilation = defaultArg dilation 1
        checkCanConv1d a.shape b.shape stride padding dilation
        let mutable b = b
        if dilation > 1 then
            b <- b.dilate([|1;1;dilation|])
        let inline fRaw(a:RawTensor,b) = a.Conv1D(b, stride, padding)
        let inline fTensor(a:Tensor,b) = a.conv1d(b, stride, padding)
        let inline dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp,bd) = ad.conv1d(bp, stride, padding) + ap.conv1d(bd, stride, padding)
        let inline dfTensorFwdTC(cp,ap,ad:Tensor) = ad.conv1d(b, stride, padding)
        let inline dfTensorFwdCT(cp,bp,bd) = a.conv1d(bd, stride, padding)
        let inline dfTensorRevTT(a,b) = Conv1DTT(a,b, stride, padding)
        let inline dfTensorRevTC(a,b) = Conv1DTTConst(a,b, stride, padding)
        let inline dfTensorRevCT(a,b) = Conv1DTConstT(a,b, stride, padding)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    // a: input, NxCxI (batchSize x inputChannels x inputLength)
    // b: filters, KxCxF (outputChannels x inputChannels x kernelLength)
    // t: output, NxKxL (batchSize x outputChannels x outputLength)
    member internal t.conv1dReverseDiff(a: Tensor, b:Tensor, aConst:bool, bConst:bool, stride:int, padding:int) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = t.shape.[0]
        let outputChannels = t.shape.[1]
        let outputLength = t.shape.[2]
        let inputChannels = a.shape.[1]
        let inputLength = a.shape.[2]
        let kernelLength = b.shape.[2]
        let mutable tderivative = t.derivative
        if stride > 1 then
            tderivative <- tderivative.dilate([|1;1;stride|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            aderivative <- a.zerosLike()
            let bFlipped = b.flip([|2|])
            for k=0 to outputChannels-1 do
                let b = bFlipped.[k].view([|inputChannels; 1; kernelLength|])
                let dBounds = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.shape.[2]-1; 1]]
                let d = tderivative.GetSlice(dBounds).view([|batchSize; 1; -1|])
                let mutable c = d.conv1d(b, padding=kernelLength-1)
                if padding > 0 then
                    let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding; c.shape.[2]-1-padding; 1]]
                    c <- c.GetSlice(cBounds).view([|batchSize; inputChannels; -1|])
                aderivative <- aderivative + c
        if not bConst then
            // propagate to b
            bderivative <- b.zerosLike()
            for n=0 to batchSize-1 do
                let aa = a.[n].view([|inputChannels; 1; inputLength|]) // treat size-one batch of a c-channel image as a size-c batch of one-channel images
                let d = tderivative.[n]
                for k=0 to outputChannels-1 do
                    let dd = d.[k].view([|1; 1; tderivative.shape.[2]|])
                    let c = aa.conv1d(dd, padding=padding).view([|1; inputChannels; kernelLength|])
                    bderivative <- bderivative.addSlice([|k; 0; 0|], c)
        aderivative, bderivative

    member a.conv2d(b:Tensor, ?stride:seq<int>, ?padding:seq<int>, ?dilation:seq<int>) =
        let stride = defaultArg stride (seq [1; 1]) |> Array.ofSeq
        let padding = defaultArg padding (seq [0; 0]) |> Array.ofSeq
        let dilation = defaultArg dilation (seq [1; 1]) |> Array.ofSeq
        checkCanConv2d a.shape b.shape stride padding dilation
        let mutable b = b
        if dilation.[0] > 1 || dilation.[1] > 1 then
            b <- b.dilate([|1; 1; dilation.[0]; dilation.[1]|])
        let inline fRaw(a:RawTensor,b) = a.Conv2D(b, stride, padding)
        let inline fTensor(a:Tensor,b) = a.conv2d(b, stride, padding)
        let inline dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp,bd) = ad.conv2d(bp, stride, padding) + ap.conv2d(bd, stride, padding)
        let inline dfTensorFwdTC(cp,ap,ad:Tensor) = ad.conv2d(b, stride, padding)
        let inline dfTensorFwdCT(cp,bp,bd) = a.conv2d(bd, stride, padding)
        let inline dfTensorRevTT(a,b) = Conv2DTT(a,b, stride, padding)
        let inline dfTensorRevTC(a,b) = Conv2DTTConst(a,b, stride, padding)
        let inline dfTensorRevCT(a,b) = Conv2DTConstT(a,b, stride, padding)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    member a.conv2d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int) =
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let dilation = defaultArg dilation 1
        a.conv2d(b, [|stride; stride|], [|padding; padding|], [|dilation; dilation|])

    member a.conv2d(b:Tensor) = a.conv2d(b, [1; 1], [0; 0], [1; 1])

    // a: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
    // b: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
    // t: output, NxKxLxM (batchSize x outputChannels x outputHeight x outputLength)
    member internal t.conv2dReverseDiff(a: Tensor, b:Tensor, aConst:bool, bConst:bool, stride:int[], padding:int[]) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = t.shape.[0]
        let outputChannels = t.shape.[1]
        // let outputHeight = t.shape.[2]
        // let outputWidth = t.shape.[3]
        let inputChannels = a.shape.[1]
        let inputHeight = a.shape.[2]
        let inputWidth = a.shape.[3]
        let kernelHeight = b.shape.[2]
        let kernelWidth = b.shape.[3]
        let mutable tderivative = t.derivative
        if stride.[0] > 1 || stride.[1] > 1 then
            tderivative <- tderivative.dilate([|1;1;stride.[0];stride.[1]|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            aderivative <- a.zerosLike()
            let bFlipped = b.flip([|2;3|])
            for k=0 to outputChannels-1 do
                let b = bFlipped.[k].view([|inputChannels; 1; kernelHeight; kernelWidth|])
                let dBounds = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.shape.[2]-1; 1]; [0; tderivative.shape.[3]-1; 1]]
                let d = tderivative.GetSlice(dBounds).view([|batchSize; 1; tderivative.shape.[2]; tderivative.shape.[3]|])
                let mutable c : Tensor = d.conv2d(b, padding=[|kernelHeight-1; kernelWidth-1|])
                if padding.[0] > 0 || padding.[1] > 0 then
                    let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding.[0]; c.shape.[2]-1-padding.[0]; 1]; [padding.[1]; c.shape.[3]-1-padding.[1]; 1]]
                    c <- c.GetSlice(cBounds).view([|batchSize; inputChannels; c.shape.[2]-2*padding.[0]; c.shape.[3]-2*padding.[1]|])
                aderivative <- aderivative + c
        if not bConst then
            // propagate to b
            bderivative <- b.zerosLike()
            for n=0 to batchSize-1 do
                let aa = a.primal.[n].view([|inputChannels; 1; inputHeight; inputWidth|]) // treat size-one batch of a c-channel image as a size-c batch of one-channel images
                let d = tderivative.[n]
                for k=0 to outputChannels-1 do
                    let dd = d.[k].view([|1; 1; tderivative.shape.[2]; tderivative.shape.[3]|])
                    let c = aa.conv2d(dd, padding=padding).view([|1; inputChannels; kernelHeight; kernelWidth|])
                    bderivative <- bderivative.addSlice([|k; 0; 0; 0|], c)
        aderivative, bderivative

    member t.reverse(?value:Tensor, ?zeroDerivatives:bool) =
        let value = defaultArg value (t.onesLike())
        let zeroDerivatives = defaultArg zeroDerivatives true
        if value.shape <> t.shape then failwithf "Expecting an adjoint value of shape %A, but received of shape %A" t.shape value.shape
        t.reverseReset(zeroDerivatives)
        t.reversePush(value)

    member inline t.backward(value) = t.reverse(value)

    member t.reverseReset(zeroDerivatives:bool) =
        let rec reset (ts: Tensor list) =
            match ts with
            | [] -> ()
            | t :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    if zeroDerivatives then t.derivative <- t.zeroLike()
                    t.fanout <- t.fanout + 1u
                    if t.fanout = 1u then
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
                        | Conv2DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv2DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv2DTConstT(_,b,_,_) -> reset (b::tt)
                        | NegT(a) -> reset (a::tt)
                        | SumT(a) -> reset (a::tt)
                        | SumT2Dim0(a) -> reset (a::tt)
                        | ExpandT(a) -> reset (a::tt)
                        | StackTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | UnstackT(a,_,_) -> reset (a::tt)
                        | CatTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | SplitT(a,_,_,_) -> reset (a::tt)
                        | TransposeT2(a) -> reset (a::tt)
                        | SqueezeT(a) -> reset (a::tt)
                        | UnsqueezeT(a) -> reset (a::tt)
                        | FlipT(a,_) -> reset (a::tt)
                        | DilateT(a,_) -> reset (a::tt)
                        | UndilateT(a,_) -> reset (a::tt)
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

    member t.reversePush(value:Tensor) =
        let rec push (ts:(Tensor*Tensor) list) =
            match ts with
            | [] -> ()
            | (v, t) :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    t.derivative <- t.derivative + v
                    t.fanout <- t.fanout - 1u
                    if t.fanout = 0u then
                        // printfn "reversepush"
                        // printfn "t %A" t
                        // printfn "o %A" o
                        match o with
                        | AddTT(a,b) -> push ((t.derivative, a) :: (t.derivative, b) :: tt)
                        | AddTTConst(a) -> push ((t.derivative, a) :: tt)
                        | AddTT0(a,b) -> push ((t.derivative, a) :: (t.derivative.sum(), b) :: tt)
                        | AddTT0Const(a) -> push ((t.derivative, a) :: tt)
                        | AddTConstT0(b) -> push ((t.derivative.sum(), b) :: tt)
                        | AddT2T1(a,b) -> push ((t.derivative, a) :: (t.derivative.sumT2Dim0(), b) :: tt)
                        | AddT2T1Const(a) -> push ((t.derivative, a) :: tt)
                        | AddT2ConstT1(b) -> push ((t.derivative.sumT2Dim0(), b) :: tt)
                        | SubTT(a,b) -> push ((t.derivative, a) :: (-t.derivative, b) :: tt)
                        | SubTTConst(a) -> push ((t.derivative, a) :: tt)
                        | SubTConstT(b) -> push ((-t.derivative, b) :: tt)
                        | SubT0T(a,b) -> push ((t.derivative.sum(), a) :: (-t.derivative, b) :: tt)
                        | SubT0TConst(a) -> push ((t.derivative.sum(), a) :: tt)
                        | SubT0ConstT(b) -> push ((-t.derivative, b) :: tt)
                        | SubTT0(a,b) -> push ((t.derivative, a) :: (-t.derivative.sum(), b) :: tt)
                        | SubTT0Const(a) -> push ((t.derivative, a) :: tt)
                        | SubTConstT0(b) -> push ((-t.derivative.sum(), b) :: tt)      
                        | MulTT(a,b) -> push ((t.derivative * b.primal, a) :: (t.derivative * a.primal, b) :: tt)
                        | MulTTConst(a,b) -> push ((t.derivative * b, a) :: tt)
                        | MulTT0(a,b) -> push ((t.derivative * b.primal, a) :: ((t.derivative * a.primal).sum(), b) :: tt)
                        | MulTConstT0(a,b) -> push (((t.derivative * a).sum(), b) :: tt)
                        | MulTT0Const(a,b) -> push ((t.derivative * b, a) :: tt)
                        | DivTT(a,b) -> push ((t.derivative / b.primal, a) :: ((t.derivative * (-a.primal / (b.primal * b.primal))), b) :: tt)
                        | DivTTConst(a,b) -> push ((t.derivative / b, a) :: tt)
                        | DivTConstT(a,b) -> push (((t.derivative * (-a / (b.primal * b.primal))), b) :: tt)
                        | DivT0T(a,b) -> push (((t.derivative / b.primal).sum(), a) :: ((t.derivative * (-a.primal / (b.primal * b.primal))), b) :: tt)
                        | DivT0TConst(a,b) -> push (((t.derivative / b).sum(), a) :: tt)
                        | DivT0ConstT(a,b) -> push (((t.derivative * (-a / (b.primal * b.primal))), b) :: tt)
                        | DivTT0(a,b) -> push ((t.derivative / b.primal, a) :: ((t.derivative * (-a.primal / (b.primal * b.primal))).sum(), b) :: tt)
                        | DivTT0Const(a,b) -> push ((t.derivative / b, a) :: tt)
                        | DivTConstT0(a,b) -> push (((t.derivative * (-a / (b.primal * b.primal))).sum(), b) :: tt)
                        | PowTT(a,b) -> push ((t.derivative * (a.primal ** (b.primal - 1.)) * b.primal, a) :: (t.derivative * (a.primal ** b.primal) * log a.primal, b) :: tt)
                        | PowTTConst(a,b) -> push ((t.derivative * (a.primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT(a,b) -> push ((t.derivative * (a ** b.primal) * log a, b) :: tt)
                        | PowT0T(a,b) -> push (((t.derivative * (a.primal ** (b.primal - 1.)) * b.primal).sum(), a) :: (t.derivative * (a.primal ** b.primal) * log a.primal, b) :: tt)
                        | PowT0TConst(a,b) -> push (((t.derivative * (a.primal ** (b - 1.)) * b).sum(), a) :: tt)
                        | PowT0ConstT(a,b) -> push ((t.derivative * (a ** b.primal) * log a, b) :: tt)
                        | PowTT0(a,b) -> push ((t.derivative * (a.primal ** (b.primal - 1.)) * b.primal, a) :: ((t.derivative * (a.primal ** b.primal) * log a.primal).sum(), b) :: tt)
                        | PowTT0Const(a,b) -> push ((t.derivative * (a.primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT0(a,b) -> push (((t.derivative * (a ** b.primal) * log a).sum(), b) :: tt)
                        | MatMulT2T2(a,b) -> push ((t.derivative.matmul(b.primal.transpose()), a) :: (a.primal.transpose().matmul(t.derivative), b) :: tt)
                        | MatMulT2T2Const(a,b) -> push ((t.derivative.matmul(b.transpose()), a) :: tt)
                        | MatMulT2ConstT2(a,b) -> push ((a.transpose().matmul(t.derivative), b) :: tt)
                        | Conv1DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = t.conv1dReverseDiff(a, b, false, false, stride, padding)
                            push ((aderivative, a) :: (bderivative, b) :: tt)
                        | Conv1DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = t.conv1dReverseDiff(a, b, false, true, stride, padding)
                            push ((aderivative, a) :: tt)                        
                        | Conv1DTConstT(a,b,stride,padding) ->
                            let _, bderivative = t.conv1dReverseDiff(a, b, true, false, stride, padding)
                            push ((bderivative, b) :: tt)                        
                        | Conv2DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = t.conv2dReverseDiff(a, b, false, false, stride, padding)
                            push ((aderivative, a) :: (bderivative, b) :: tt)
                        | Conv2DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = t.conv2dReverseDiff(a, b, false, true, stride, padding)
                            push ((aderivative, a) :: tt)
                        | Conv2DTConstT(a,b,stride,padding) ->
                            let _, bderivative = t.conv2dReverseDiff(a, b, true, false, stride, padding)
                            push ((bderivative, b) :: tt)
                        | NegT(a) -> push ((-t.derivative, a) :: tt)
                        | SumT(a) -> push ((t.derivative.expand(a.shape), a) :: tt)
                        | SumT2Dim0(a) -> push ((a.zerosLike() + t.derivative, a) :: tt)
                        | ExpandT(a) -> push ((t.derivative.sumToSize(a.shape), a) :: tt)
                        | StackTs(a,dim) ->
                            push (List.append (Array.zip (t.derivative.unstack(dim)) a |> Array.toList) tt)
                        | UnstackT(a,dim,i) -> 
                            if a.derivative.dim = 0 then a.derivative <- a.zerosLike() + a.derivative
                            a.derivative <- a.derivative.addSlice(Array.init a.dim (fun j -> if j=dim then i else 0), t.derivative.unsqueeze(dim))
                            push ((a.zeroLike(), a) :: tt)
                        | CatTs(a, dim) ->
                            let sizes = a |> Array.map (fun x -> x.shape.[dim])
                            push (List.append (Array.zip (t.derivative.split(sizes, dim=dim)) a |> Array.toList) tt)
                        | SplitT(a,sizes,dim,i) -> 
                            if a.derivative.dim = 0 then a.derivative <- a.zerosLike() + a.derivative
                            let locs = (0,sizes) ||> Array.scan (+)
                            a.derivative <- a.derivative.addSlice(Array.init a.dim (fun j -> if j=dim then locs.[i] else 0), t.derivative)
                            push ((a.zeroLike(), a) :: tt)
                        | TransposeT2(a) -> push ((t.derivative.transpose(), a) :: tt)
                        | SqueezeT(a) -> push ((t.derivative.viewAs(a), a) :: tt)
                        | UnsqueezeT(a) -> push ((t.derivative.viewAs(a), a) :: tt)
                        | FlipT(a, dims) -> push ((t.derivative.flip(dims), a) :: tt)
                        | DilateT(a, dilations) -> push ((t.derivative.undilate(dilations), a) :: tt)
                        | UndilateT(a, dilations) -> push ((t.derivative.dilate(dilations), a) :: tt)
                        | ViewT(a,aShape) -> push (((t.derivative.view(aShape)), a) :: tt)
                        | SliceT(a,bounds) -> 
                            // TODO: Tensor.ZerosLike(a) below is to handle non-scalar TensorRs with a scalar derivative Tensor(0.) (representing the initialization before accumulation). This is correct but can be changed to eliminate the extra op.
                            if a.derivative.dim = 0 then a.derivative <- a.zerosLike() + a.derivative
                            a.derivative <- a.derivative.addSlice(boundsToLocation bounds, t.derivative.view(boundsToShape bounds))
                            push ((a.zeroLike(), a) :: tt)
                        | AddTTSlice(a,location,b) -> push ((t.derivative, a) :: (t.derivative.GetSlice(shapeLocationToBounds b.shape location), b):: tt)
                        | AddTTConstSlice(a) -> push ((t.derivative, a) :: tt)
                        | AddTConstTSlice(location, b) -> push ((t.derivative.GetSlice(shapeLocationToBounds b.shape location), b):: tt)
                        | SignT(a) -> push ((a.zerosLike(), a) :: tt)
                        | FloorT(a) -> push ((a.zerosLike(), a) :: tt)
                        | CeilT(a) -> push ((a.zerosLike(), a) :: tt)
                        | RoundT(a) -> push ((a.zerosLike(), a) :: tt)
                        | AbsT(a) -> push ((t.derivative * a.primal.sign(), a) :: tt)
                        | ReluT(a) -> let sap = a.primal.sign() in push ((t.derivative * (sap.abs()) * (sap + 1.) / 2., a) :: tt)
                        | SigmoidT(a) -> push ((t.derivative * t.primal * (1. - t.primal), a) :: tt)
                        | ExpT(a) -> push ((t.derivative * t.primal, a) :: tt)
                        | LogT(a) -> push ((t.derivative / a.primal, a) :: tt)
                        | Log10T(a) -> push ((t.derivative / (a.primal * log10Val), a) :: tt)
                        | SqrtT(a) -> push ((t.derivative / (2. * t.primal), a) :: tt)
                        | SinT(a) -> push ((t.derivative * (a.primal.cos()), a) :: tt)
                        | CosT(a) -> push ((-t.derivative * (a.primal.sin()), a) :: tt)
                        | TanT(a) -> let cosap = a.primal.cos() in push ((t.derivative / (cosap * cosap), a) :: tt)
                        | SinhT(a) -> push ((t.derivative * (a.primal.cosh()), a) :: tt)
                        | CoshT(a) -> push ((t.derivative * (a.primal.sinh()), a) :: tt)
                        | TanhT(a) -> let coshap = a.primal.cosh() in push ((t.derivative / (coshap * coshap), a) :: tt)
                        | AsinT(a) -> push ((t.derivative / Tensor.Sqrt(1. - a.primal*a.primal), a) :: tt)
                        | AcosT(a) -> push ((-t.derivative / Tensor.Sqrt(1. - a.primal*a.primal), a) :: tt)
                        | AtanT(a) -> push ((t.derivative / (1. + a.primal*a.primal), a) :: tt)
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

    | Conv2DTT of Tensor * Tensor * int[] * int[]
    | Conv2DTTConst of Tensor * Tensor * int[] * int[]
    | Conv2DTConstT of Tensor * Tensor * int[] * int[]

    | NegT of Tensor
    | SumT of Tensor
    | SumT2Dim0 of Tensor
    | ExpandT of Tensor
    | StackTs of Tensor[] * dim:int
    | UnstackT of Tensor * dim:int * i:int
    | CatTs of Tensor[] * dim:int
    | SplitT of Tensor * int[] * dim:int * i:int
    | SliceT of Tensor * int[,]
    | AddTTSlice of Tensor * int[] * Tensor
    | AddTTConstSlice of Tensor
    | AddTConstTSlice of int[] * Tensor
    | TransposeT2 of Tensor
    | SqueezeT of Tensor
    | UnsqueezeT of Tensor
    | FlipT of Tensor * int[]
    | DilateT of Tensor * int[]
    | UndilateT of Tensor * int[]
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
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option) =
        // Dims: 1
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int) =
        // Dims: 1
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option) =
        // Dims: 2
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int) =
        // Dims: 2
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<System.Diagnostics.CodeAnalysis.ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)

[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("DiffSharp.Tests")>]
do()
