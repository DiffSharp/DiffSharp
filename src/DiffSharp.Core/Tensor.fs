namespace DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open System
open System.IO
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open System.Diagnostics.CodeAnalysis

#nowarn "1182" // turn off compiler-generated unused variable warnings in this file only

type scalar = IConvertible

[<CustomEquality; CustomComparison>]
type Tensor = 
    | Tensor of primalRaw:RawTensor
    | TensorF of primal:Tensor * derivative:Tensor * nestingTag:uint32
    | TensorR of primal:Tensor * derivative:(Tensor ref) * parentOp:TensorOp * fanout:(uint32 ref) * nestingTag:uint32

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

    member t.cast(dtype) =
        if t.dtype = dtype then t else
        match t with
        | Tensor(tp) -> Tensor(tp.Cast(dtype))
        | TensorF(_) -> failwith "Cannot cast TensorF - do not cast during differentiation"
        | TensorR(_) -> failwith "Cannot cast TensorR - do not cast during differentiation"

    member t.move(backend) =
        if t.backend = backend then t else
        match t with
        | Tensor(tp) -> 
            let tpflat = tp.ViewT([|tp.Nelement|]) //
            let tpflatValues = tpflat.ToValues()
            Tensor(tp.CreateLike(tpflatValues, backend=backend).ViewT(tp.Shape))
        | TensorF(_) -> failwith "Cannot move TensorF - do not move during differentiation"
        | TensorR(_) -> failwith "Cannot move TensorR - do not move during differentiation"

    member t.move(device) =
        if t.device = device then t else
        match t with
        | Tensor(tp) -> Tensor(tp.MoveTo(device))
        | TensorF(_) -> failwith "Cannot move TensorF - do not move during differentiation"
        | TensorR(_) -> failwith "Cannot move TensorR - do not move during differentiation"

    member t.move(?dtype:Dtype, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        let backend = defaultArg backend Backend.Default
        t.cast(dtype).move(device).move(backend)

    member internal t.castAfterSummation(?dtype:Dtype) =
        match dtype with
        | None -> t
        | Some dt -> t.cast(dt)

    member t.cpu() = t.move(Device.CPU)
    member t.gpu() = t.move(Device.GPU)

    member t.bool() = t.cast(Dtype.Bool)
    member t.int8() = t.cast(Dtype.Int8)
    member t.int16() = t.cast(Dtype.Int16)
    member t.int32() = t.cast(Dtype.Int32)
    member t.int() = t.cast(Dtype.Int32)
    member t.int64() = t.cast(Dtype.Int64)
    member t.float32() = t.cast(Dtype.Float32)
    member t.float64() = t.cast(Dtype.Float64)
    member t.float() = t.cast(Dtype.Float64)
    member t.double() = t.cast(Dtype.Float64)

    member t.dtype = t.primalRaw.Dtype
    member t.device = t.primalRaw.Device
    member t.backend = t.primalRaw.Backend

    member t.depth =
        let rec depth x d =
            match x with
            | Tensor(_) -> d
            | TensorF(tp,_,_) -> depth tp (d + 1)
            | TensorR(tp,_,_,_,_) -> depth tp (d + 1)
        depth t 0

    member t.parentOp =
        match t with
        | Tensor(_) -> failwith "Cannot get derivative of constant Tensor"
        | TensorF(_)-> failwith "Cannot get parent operation of TensorF"
        | TensorR(_,_,o,_,_) -> o

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
    member t.isForwardDiff() =
        match t with
        | TensorF(_) -> true
        | _ -> false
    member t.isReverseDiff() =
        match t with
        | TensorR(_) -> true
        | _ -> false
    member t.isNoDiff() =
        match t with
        | Tensor(_) -> true
        | _ -> false
    member t.shape = t.primalRaw.Shape
    member t.dim = t.primalRaw.Dim
    member t.nelement = t.primalRaw.Nelement
    member t.toArray() = t.primalRaw.ToArray()
    member t.toScalar() = t.primalRaw.ToScalar()
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

    member t.save(fileName:string) = saveBinary t fileName
    static member load(fileName:string):Tensor = loadBinary fileName

    member t.summary() =
        match t with
        | Tensor(_) -> sprintf "Tensor %A" t.shape
        | TensorF(_) -> sprintf "TensorF %A" t.shape
        | TensorR(_,_,o,_,_) -> 
            let c, _ = Reflection.FSharpValue.GetUnionFields(o, typeof<TensorOp>)
            let fields = c.GetFields()
            sprintf "TensorR %A %s" t.shape c.Name

    member t.parents() =
        let mutable p = []
        let rec parents (t:obj) d =
            match t with
            | :? Tensor as t ->
                p <- p |> List.append [t]
                match t with
                | Tensor(_) -> sprintf "Tensor %A" t.shape
                | TensorF(_) -> sprintf "TensorF %A" t.shape
                | TensorR(_,_,o,_,_) -> 
                    let c, _ = Reflection.FSharpValue.GetUnionFields(o, typeof<TensorOp>)
                    let fields = c.GetFields()
                    let mutable ret = sprintf "TensorR %A %s" t.shape c.Name
                    for field in fields do
                        let fv = field.GetValue(o)
                        ret <- ret + sprintf "\n%s%s" (String.replicate d " ") (parents fv (d+1))
                    ret
            | :? (Tensor array) as ts ->
                // p <- p |> List.append (ts |> Array.toList)
                let mutable ret = ""
                let mutable prefix = ""
                for t in ts do
                    ret <- ret + sprintf "%s%s%s" prefix (String.replicate d " ") (parents t (d+1))
                    prefix <- "\n"
                ret
            | _ -> indentNewLines (sprintf "%A" t) d
        let ps = parents t 1
        p |> List.rev, ps

    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.primalRaw.Equals(tensor.primalRaw)
        | _ -> false
    override t.GetHashCode() = hash t.primalRaw
    interface System.IComparable with
        override t.CompareTo(other) =
            match other with
            | :? Tensor as tensor -> 
                if t.dim = tensor.dim && t.dim = 0 then
                    (t.primalRaw :> System.IComparable).CompareTo(tensor.primalRaw)
                else
                    failwith "Cannot compare non-scalar Tensors"
            | _ -> failwith "Cannot compare Tensor with another type"

    static member op_Explicit(tensor:Tensor):single = tensor.toScalar() |> Convert.ToSingle
    static member op_Explicit(tensor:Tensor):double = tensor.toScalar() |> Convert.ToDouble
    static member op_Explicit(tensor:Tensor):byte = tensor.toScalar() |> Convert.ToByte
    static member op_Explicit(tensor:Tensor):int8 = tensor.toScalar() |> Convert.ToSByte
    static member op_Explicit(tensor:Tensor):int16 = tensor.toScalar() |> Convert.ToInt16
    static member op_Explicit(tensor:Tensor):int32 = tensor.toScalar() |> Convert.ToInt32
    static member op_Explicit(tensor:Tensor):int64 = tensor.toScalar() |> Convert.ToInt64
    static member op_Explicit(tensor:Tensor):bool = tensor.toScalar() |> Convert.ToBoolean

    interface System.IConvertible with
        override t.GetTypeCode() = TypeCode.Object
        override t.ToSingle(_) = Tensor.op_Explicit(t)
        override t.ToDouble(_) = Tensor.op_Explicit(t)
        override t.ToByte(_) = Tensor.op_Explicit(t)
        override t.ToSByte(_) = Tensor.op_Explicit(t)
        override t.ToInt16(_) = Tensor.op_Explicit(t)
        override t.ToInt32(_) = Tensor.op_Explicit(t)
        override t.ToInt64(_) = Tensor.op_Explicit(t)
        override t.ToBoolean(_) = Tensor.op_Explicit(t)
        override t.ToChar(_) = failwithf "Cannot convert Tensor to Char"
        override t.ToDateTime(_) = failwithf "Cannot convert Tensor to DateTime"
        override t.ToDecimal(_) = failwithf "Cannot convert Tensor to Decimal"
        override t.ToString(_) = failwithf "Cannot convert Tensor to String"
        override t.ToType(_,_) = failwithf "Cannot convert Tensor to Type"
        override t.ToUInt16(_) = failwithf "Cannot convert Tensor to UInt16"
        override t.ToUInt32(_) = failwithf "Cannot convert Tensor to UInt32"
        override t.ToUInt64(_) = failwithf "Cannot convert Tensor to UInt64"

    member t.allclose(tensor:Tensor, ?relativeTolerance, ?absoluteTolerance) =
        let relativeTolerance = defaultArg relativeTolerance 1e-5
        let absoluteTolerance = defaultArg absoluteTolerance 1e-8
        t.primalRaw.AllClose(tensor.primalRaw, relativeTolerance, absoluteTolerance)

    member a.zerosLike(?shape:seq<int>, ?dtype, ?device, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.ZerosLike(shape |> Array.ofSeq, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.onesLike(?shape:seq<int>, ?dtype, ?device, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.OnesLike(shape |> Array.ofSeq, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.fullLike(shape:seq<int>, value:scalar, ?dtype, ?device, ?backend) = 
        Tensor(a.primalRaw.FullLike(shape |> Array.ofSeq, value, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.scalarLike(scalar:IConvertible, ?dtype, ?device, ?backend) = 
        a.fullLike([], scalar, ?dtype=dtype, ?device=device, ?backend=backend)
    member a.randLike(?shape:seq<int>, ?dtype, ?device, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.RandomLike((shape |> Array.ofSeq), ?dtype=dtype, ?device=device, ?backend=backend))
    member a.randnLike(?shape:seq<int>, ?dtype, ?device, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.RandomNormalLike(shape |> Array.ofSeq, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.randintLike(low:int, high:int, ?shape:seq<int>, ?dtype, ?device, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        Tensor(a.primalRaw.RandomIntLike(shape |> Array.ofSeq, low, high, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.zeroLike(?dtype, ?device, ?backend) = Tensor(a.primalRaw.ZeroLike(?dtype=dtype, ?device=device, ?backend=backend))
    member a.oneLike(?dtype, ?device, ?backend) = Tensor(a.primalRaw.OneLike(?dtype=dtype, ?device=device, ?backend=backend))
    member a.arangeLike(endVal:float, ?startVal:float, ?step:float, ?dtype, ?device, ?backend) =
        let startVal = defaultArg startVal 0.
        let step = defaultArg step 1.
        let length = (endVal - startVal) / step |> ceil |> int
        let v = Array.init length (fun i -> startVal + float(i) * step)
        a.like(box v, ?dtype=dtype, ?device=device, ?backend=backend)
    member a.arangeLike(endVal:int, ?startVal:int, ?step:int, ?dtype, ?device, ?backend) =
        let endVal = endVal |> float
        let startVal = defaultArg startVal 0 |> float
        let step = defaultArg step 1 |> float
        let dtype = defaultArg dtype Dtype.Int32
        a.arangeLike(endVal=endVal, startVal=startVal, step=step, dtype=dtype, ?device=device, ?backend=backend)
    member a.like(value, ?dtype, ?device, ?backend) = Tensor(a.primalRaw.CreateLike(value, ?dtype=dtype, ?device=device, ?backend=backend))
    member a.clone() = Tensor(a.primalRaw.Clone())
    member a.onehotLike(length:int, hot:int, ?dtype, ?device, ?backend) =
        if hot < 0 || hot >= length then failwithf "Expecting 0 <= hot < length"
        a.zerosLike([|length|], ?dtype=dtype, ?device=device, ?backend=backend).addSlice([|hot|], a.onesLike([|1|], ?dtype=dtype, ?device=device, ?backend=backend))
    member a.lt(b:Tensor) = Tensor(a.primalRaw.LtTT(b.primalRaw))
    member a.gt(b:Tensor) = Tensor(a.primalRaw.GtTT(b.primalRaw))
    member a.le(b:Tensor) =Tensor(a.primalRaw.LeTT(b.primalRaw))
    member a.ge(b:Tensor) = Tensor(a.primalRaw.GeTT(b.primalRaw))
    member a.isinf() = Tensor(a.primalRaw.IsInfT())
    member a.isnan() = Tensor(a.primalRaw.IsNaNT())
    member a.hasinf() = a.isinf().sum() > a.zeroLike(dtype=Dtype.Int64)
    member a.hasnan() = a.isnan().sum() > a.zeroLike(dtype=Dtype.Int64)
    member a.maxIndex() = a.primalRaw.MaxIndexT()
    member a.minIndex() = a.primalRaw.MinIndexT()
    member a.max() = a.[a.maxIndex()]
    member a.min() = a.[a.minIndex()]
    member a.max(b:Tensor) = ((a + b) + Tensor.Abs(b - a)) / 2.
    member a.max(b) = a.max(a.like(b))
    member a.min(b:Tensor) = ((a + b) - Tensor.Abs(a - b)) / 2.
    member a.min(b) = a.min(a.like(b))

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

    static member create(value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        let res = value |> tryFlatArrayAndShape<Tensor> // support creation of new Tensor from a structure holding scalar Tensors
        match res with
        | Some (array, shape) -> 
            let array = array |> Array.map float32
            let value = arrayND shape (fun ii -> array.[indexToFlatIndex shape ii])
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))
        | None ->
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))        

    static member multinomial(probs:Tensor, numSamples:int) =
        if probs.dim < 1 || probs.dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" probs.shape
        if probs.dim = 1 then
            let p = 
                match probs.dtype with
                | Dtype.Float32 -> probs.toArray() :?> float32[] |> Array.map Convert.ToDouble
                | Dtype.Float64 -> probs.toArray() :?> float[]
                | _ -> failwithf "Expecting probs to have dtype Float32 or Float64, received %A" probs.dtype
            Tensor.create(Random.Multinomial(p, numSamples), dtype=Dtype.Int32)
        else
            let p = 
                match probs.dtype with
                | Dtype.Float32 -> probs.toArray() :?> float32[,] |> Array2D.map Convert.ToDouble
                | Dtype.Float64 -> probs.toArray() :?> float[,]
                | _ -> failwithf "Expecting probs to have dtype Float32 or Float64, received %A" probs.dtype
            Tensor.create(Random.Multinomial(p, numSamples), dtype=Dtype.Int32)

    static member stack(tensors:seq<Tensor>, ?dim:int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        // TODO: check if all Tensors are of the same type (Tensor, TensorF, or TensorR) and have the same nesting tag
        let shapes = tensors |> Array.map (fun t -> t.shape)
        Shape.checkCanStack shapes dim |> ignore
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
        Shape.checkCanUnstack a.shape |> ignore
        match a with
        | Tensor(ap) -> ap.UnstackT(dim) |> Array.map Tensor
        | TensorF(ap,ad,at) -> Array.map2 (fun p d -> TensorF(p,d,at)) (ap.unstack(dim)) (ad.unstack(dim))
        | TensorR(ap,_,_,_,at) -> Array.mapi (fun i p -> TensorR(p, ref (p.zeroLike()), UnstackT(a, dim, i), ref 0u, at)) (ap.unstack(dim))

    static member cat(tensors:seq<Tensor>, ?dim: int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        // TODO: check if all Tensors are of the same nesting variety (Tensor, TensorF, or TensorR), have the same nesting tag, and have the same dtype, device, backend
        match Seq.head tensors with
        | Tensor(ap) -> Tensor(ap.CatTs((tensors |> Array.map (fun t -> t.primalRaw)), dim))
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

    static member inline (-->) (t:Tensor, f:Tensor -> ^a) = f t

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
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "+" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast + bCast
        elif a.shape = b.shape then
            let fRaw(a:RawTensor,b) = a.AddTT(b)
            let fTensor(a,b) = a + b
            let dfTensorFwdTT(cp,ap,ad,bp:Tensor,bd:Tensor) = ad + bd
            let dfTensorFwdTC(cp,ap,ad) = ad
            let dfTensorFwdCT(cp,bp,bd) = bd
            let dfTensorRevTT(a,b) = AddTT(a,b)
            let dfTensorRevTC(a,b) = AddTTConst(a)
            let dfTensorRevCT(a,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let fRaw(a,b:RawTensor) = b.AddTT0(a)
            let fTensor(a,b) = a + b
            let dfTensorFwdTT(cp,ap,ad,bp:Tensor,bd:Tensor) = ad + bd
            let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.expand(b.shape)
            let dfTensorFwdCT(cp,bp,bd) = bd
            let dfTensorRevTT(a,b) = AddTT0(b,a)
            let dfTensorRevTC(a,b) = AddTConstT0(a)
            let dfTensorRevCT(a,b) = AddTT0Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let fRaw(a:RawTensor,b) = a.AddTT0(b)
            let fTensor(a,b) = a + b
            let dfTensorFwdTT(cp,ap,ad,bp:Tensor,bd:Tensor) = ad + bd
            let dfTensorFwdTC(cp,ap,ad) = ad
            let dfTensorFwdCT(cp,bp,bd:Tensor) = bd.expand(a.shape)
            let dfTensorRevTT(a,b) = AddTT0(a,b)
            let dfTensorRevTC(a,b) = AddTT0Const(a)
            let dfTensorRevCT(a,b) = AddTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 2 && b.dim = 1 && a.shape.[1] = b.shape.[0] then
            let fRaw(a:RawTensor,b) = a.AddT2T1(b)
            let fTensor(a,b) = a + b
            let dfTensorFwdTT(cp,ap,ad,bp:Tensor,bd:Tensor) = ad + bd
            let dfTensorFwdTC(cp,ap,ad) = ad
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = cp.zerosLike() + bd
            let dfTensorRevTT(a,b) = AddT2T1(a,b)
            let dfTensorRevTC(a,b) = AddT2T1Const(a)
            let dfTensorRevCT(a,b) = AddT2ConstT1(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 1 && b.dim = 2 && a.shape.[0] = b.shape.[1] then
            let fRaw(a,b:RawTensor) = b.AddT2T1(a)
            let fTensor(a,b) = a + b
            let dfTensorFwdTT(cp,ap,ad,bp:Tensor,bd:Tensor) = ad + bd
            let dfTensorFwdTC(cp:Tensor,ap,ad) = ad + cp.zerosLike()
            let dfTensorFwdCT(cp,bp,bd) = bd
            let dfTensorRevTT(a,b) = AddT2T1(b,a)
            let dfTensorRevTC(a,b) = AddT2ConstT1(a)
            let dfTensorRevCT(a,b) = AddT2T1Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded + bExpanded
    static member (+) (a:Tensor, b) = a + a.scalarLike(b)
    static member (+) (a, b:Tensor) = b.scalarLike(a) + b
    member a.add(b:Tensor) = a + b
    member a.add(b) = a + a.scalarLike(b)

    static member (-) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "-" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast - bCast
        elif a.shape = b.shape then
            let fRaw(a:RawTensor,b) = a.SubTT(b)
            let fTensor(a,b) = a - b
            let dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let dfTensorFwdTC(cp,ap,ad) = ad
            let dfTensorFwdCT(cp,bp,bd) = -bd
            let dfTensorRevTT(a,b) = SubTT(a,b)
            let dfTensorRevTC(a,b) = SubTTConst(a)
            let dfTensorRevCT(a,b) = SubTConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let fRaw(a:RawTensor,b) = a.SubT0T(b)
            let fTensor(a,b) = a - b
            let dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.expand(b.shape)
            let dfTensorFwdCT(cp,bp,bd) = -bd
            let dfTensorRevTT(a,b) = SubT0T(a,b)
            let dfTensorRevTC(a,b) = SubT0TConst(a)
            let dfTensorRevCT(a,b) = SubT0ConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let fRaw(a:RawTensor,b) = a.SubTT0(b)
            let fTensor(a,b) = a - b
            let dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let dfTensorFwdTC(cp,ap,ad) = ad
            let dfTensorFwdCT(cp,bp,bd:Tensor) = (-bd).expand(a.shape)
            let dfTensorRevTT(a,b) = SubTT0(a,b)
            let dfTensorRevTC(a,b) = SubTT0Const(a)
            let dfTensorRevCT(a,b) = SubTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded - bExpanded
    static member (-) (a:Tensor, b) = a - a.scalarLike(b)
    static member (-) (a, b:Tensor) = b.scalarLike(a) - b
    member a.sub(b:Tensor) = a - b
    member a.sub(b) = a - a.scalarLike(b)

    static member (*) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "*" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast * bCast
        elif a.shape = b.shape then
            let fRaw(a:RawTensor,b) = a.MulTT(b)
            let fTensor(a,b) = a * b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let dfTensorRevTT(a,b) = MulTT(a,b)
            let dfTensorRevTC(a,b) = MulTTConst(a,b)
            let dfTensorRevCT(a,b) = MulTTConst(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let fRaw(a,b:RawTensor) = b.MulTT0(a)
            let fTensor(a,b) = a * b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let dfTensorRevTT(a,b) = MulTT0(b,a)
            let dfTensorRevTC(a,b) = MulTConstT0(b,a)
            let dfTensorRevCT(a,b) = MulTT0Const(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let fRaw(a:RawTensor,b) = a.MulTT0(b)
            let fTensor(a,b) = a * b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad * bp) + (ap * bd)
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad * b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = a * bd
            let dfTensorRevTT(a,b) = MulTT0(a,b)
            let dfTensorRevTC(a,b) = MulTT0Const(a,b)
            let dfTensorRevCT(a,b) = MulTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded * bExpanded
    static member (*) (a:Tensor, b) = a * a.scalarLike(b)
    static member (*) (a, b:Tensor) = b.scalarLike(a) * b
    member a.mul(b:Tensor) = a * b
    member a.mul(b) = a * a.scalarLike(b)

    static member (/) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "/" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast / bCast
        elif a.shape = b.shape then
            let fRaw(a:RawTensor,b) = a.DivTT(b)
            let fTensor(a,b) = a / b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let dfTensorRevTT(a,b) = DivTT(a,b)
            let dfTensorRevTC(a,b) = DivTTConst(a,b)
            let dfTensorRevCT(a,b) = DivTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let fRaw(a:RawTensor,b) = a.DivT0T(b)
            let fTensor(a,b) = a / b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let dfTensorRevTT(a,b) = DivT0T(a,b)
            let dfTensorRevTC(a,b) = DivT0TConst(a,b)
            let dfTensorRevCT(a,b) = DivT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let fRaw(a:RawTensor,b) = a.DivTT0(b)
            let fTensor(a:Tensor,b:Tensor) = a / b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ad - bd * cp) / bp
            let dfTensorFwdTC(cp:Tensor,ap:Tensor,ad:Tensor) = ad / b
            let dfTensorFwdCT(cp:Tensor,bp:Tensor,bd:Tensor) = -bd * cp / bp
            let dfTensorRevTT(a,b) = DivTT0(a,b)
            let dfTensorRevTC(a,b) = DivTT0Const(a,b)
            let dfTensorRevCT(a,b) = DivTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded / bExpanded
    static member (/) (a:Tensor, b) = a / a.scalarLike(b)
    static member (/) (a, b:Tensor) = b.scalarLike(a) / b
    member a.div(b:Tensor) = a / b
    member a.div(b) = a / a.scalarLike(b)

    static member Pow (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "Pow" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                Tensor.Pow (aCast, bCast)
        elif a.shape = b.shape then
            let fRaw(a:RawTensor,b) = a.PowTT(b)
            let fTensor(a:Tensor,b:Tensor) = a ** b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let dfTensorRevTT(a,b) = PowTT(a,b)
            let dfTensorRevTC(a,b) = PowTTConst(a,b)
            let dfTensorRevCT(a,b) = PowTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.dim = 0 then
            let fRaw(a:RawTensor,b) = a.PowT0T(b)
            let fTensor(a:Tensor,b:Tensor) = a ** b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let dfTensorRevTT(a,b) = PowT0T(a,b)
            let dfTensorRevTC(a,b) = PowT0TConst(a,b)
            let dfTensorRevCT(a,b) = PowT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.dim = 0 then
            let fRaw(a:RawTensor,b) = a.PowTT0(b)
            let fTensor(a:Tensor,b:Tensor) = a ** b
            let dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let dfTensorRevTT(a,b) = PowTT0(a,b)
            let dfTensorRevTC(a,b) = PowTT0Const(a,b)
            let dfTensorRevCT(a,b) = PowTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            Tensor.Pow(aExpanded, bExpanded)

    static member Pow (a:Tensor, b:float) = a ** a.scalarLike(b)
    static member Pow (a:Tensor, b:int) = a ** a.scalarLike(b)
    static member Pow (a:Tensor, b) = a ** a.scalarLike(b)
    static member Pow (a:float, b:Tensor) = b.scalarLike(a) ** b
    static member Pow (a:int, b:Tensor) = b.scalarLike(a) ** b
    static member Pow (a, b:Tensor) = b.scalarLike(a) ** b
    member a.pow(b:Tensor) = a ** b
    member a.pow(b) = a ** a.scalarLike(b)

    member a.matmul (b:Tensor) =
        Shape.checkCanMatmul a.shape b.shape
        let fRaw(a:RawTensor,b) = a.MatMulT2T2(b)
        let fTensor(a:Tensor,b) = a.matmul(b)
        let dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = ad.matmul(bp) + ap.matmul(bd)
        let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.matmul(b)
        let dfTensorFwdCT(cp,bp,bd) = a.matmul(bd)
        let dfTensorRevTT(a,b) = MatMulT2T2(a,b)
        let dfTensorRevTC(a,b) = MatMulT2T2Const(a,b)
        let dfTensorRevCT(a,b) = MatMulT2ConstT2(a,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    member a.dot(b:Tensor) =
        Shape.checkCanDot a.shape b.shape
        let a:Tensor = a.view([1;a.nelement])
        let b:Tensor = b.view([b.nelement;1])
        a.matmul(b).view([])

    static member (~-) (a:Tensor) =
        let fRaw(a:RawTensor) = a.NegT()
        let fTensor(a) = -a
        let dfTensorFwd(cp,ap,ad) = -ad
        let dfTensorRev(a) = NegT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member a.neg() = -a

    member a.sum(?dtype: Dtype) =
        let fRaw(a:RawTensor) = a.SumT(?resultType=dtype)
        let fTensor(a:Tensor) = a.sum(?dtype=dtype)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.sum(?dtype=dtype)
        let dfTensorRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    // TODO: this can be implemented in a more memory efficient way by pushing the sum operation to the RawTensor level and implementing the derivatives using general broadcasting when it's available
    member a.sum(dim:int, ?keepDim:bool, ?dtype: Dtype) =
       let keepDim = defaultArg keepDim false
       let res =
        if dim = 0 && a.dim = 0 then a
        else
            if dim >= a.dim || dim < 0 then failwithf "Expecting dim to be between 0 and %A" a.dim
            let sBounds = Array2D.init a.dim 3 (fun i j -> if j=0 then 0 elif j=1 then a.shape.[i]-1 else 0)
            sBounds.[dim, 1] <- 0
            sBounds.[dim, 2] <- 1
            let mutable s = a.zerosLike(dtype=a.dtype.SummationType).GetSlice(sBounds)
            for i=0 to a.shape.[dim]-1 do
                sBounds.[dim,0] <- i
                sBounds.[dim,1] <- i
                sBounds.[dim,2] <- 1
                s <- s + a.GetSlice(sBounds).cast(a.dtype.SummationType)
            s
       let res2 = if keepDim then res.unsqueeze(dim) else res
       res2.castAfterSummation(?dtype=dtype)

    /// Reduce the dimensionality via summation until we reach `newShape`.  An expansion
    /// from newShape to shape must be possible.
    member a.sumToSize(newShape:int[], ?dtype: Dtype) =
        let oldShape = a.shape
        if oldShape = newShape then
            a.cast(defaultArg dtype a.dtype.SummationType)
        elif newShape.Length = 0 then
            a.sum(?dtype=dtype)
        else
            Shape.checkCanExpand newShape oldShape
            let trim = oldShape.Length - newShape.Length
            let mutable result = a.cast(a.dtype.SummationType)
            // collapse the eliminated dimensions
            for _dim in 0 .. trim-1 do 
                result <- result.sum(0, keepDim=false)
            // reduce the squeezed dimensions
            for dim in 0 .. newShape.Length-1 do 
                if oldShape.[trim+dim] <> newShape.[dim] then 
                    result <- result.sum(dim, keepDim=true)
            result.castAfterSummation(?dtype=dtype)

    member a.mean() = a.sum() / a.nelement

    member a.mean(dim:int, ?keepDim:bool) = 
        if dim = 0 && a.dim = 0 then a
        else 
           let sm = a.sum(dim, ?keepDim=keepDim)
           let dv = sm / a.shape.[dim]
           dv

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
        let fRaw(a:RawTensor) = a.SumT2Dim0()
        let fTensor(a:Tensor) = a.sumT2Dim0()
        let dfTensorFwd(cp,ap,ad:Tensor):Tensor = ad.sumT2Dim0()
        let dfTensorRev(a) = SumT2Dim0(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    
    member a.transpose() =
        Shape.checkCanTranspose a.dim
        let fRaw(a:RawTensor) = a.TransposeT2()
        let fTensor(a:Tensor) = a.transpose()
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.transpose()
        let dfTensorRev(a) = TransposeT2(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.squeeze(?dim:int) =
        let dim = defaultArg dim -1
        let fRaw(a:RawTensor) = a.SqueezeT(dim)
        let fTensor(a:Tensor) = a.squeeze(dim)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.squeeze(dim)
        let dfTensorRev(a) = SqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.unsqueeze(dim:int) =
        let fRaw(a:RawTensor) = a.UnsqueezeT(dim)
        let fTensor(a:Tensor) = a.unsqueeze(dim)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.unsqueeze(dim)
        let dfTensorRev(a) = UnsqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.flip(dims:seq<int>) =
        let dims = dims |> Array.ofSeq
        Shape.checkCanFlip a.dim dims
        let fRaw(a:RawTensor) = a.FlipT(dims)
        let fTensor(a:Tensor) = a.flip(dims)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.flip(dims)
        let dfTensorRev(a) = FlipT(a, dims)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.dilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        Shape.checkCanDilate a.dim dilations
        let fRaw(a:RawTensor) = a.DilateT(dilations)
        let fTensor(a:Tensor) = a.dilate(dilations)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.dilate(dilations)
        let dfTensorRev(a) = DilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.undilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        let fRaw(a:RawTensor) = a.UndilateT(dilations)
        let fTensor(a:Tensor) = a.undilate(dilations)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.undilate(dilations)
        let dfTensorRev(a) = UndilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.repeat(dim:int, times:int) =
        Shape.checkCanRepeat a.shape dim
        let newShape = a.shape |> Array.copy
        newShape.[dim] <- times
        let mutable ret = a.zerosLike(newShape)
        let location = Array.create a.dim 0
        for i=0 to times-1 do
            location.[dim] <- i
            ret <- ret.addSlice(location, a)
        ret

    member a.gather(dim:int, indices:Tensor) =
        Shape.checkCanGather a.shape dim indices.shape indices.dtype
        let fRaw(a:RawTensor) = a.GatherT(dim, indices.primalRaw)
        let fTensor(a:Tensor) = a.gather(dim, indices)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.gather(dim, indices)
        let dfTensorRev(a) = GatherT(a, dim, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.view(shape:seq<int>) =
        let shape = shape |> Seq.toArray |> Shape.complete a.nelement  // Handles -1 semantics
        Shape.checkCanView a.shape shape
        let fRaw(a:RawTensor) = a.ViewT(shape)
        let fTensor(a:Tensor) = a.view(shape)
        let dfTensorFwd(cp,ap,ad:Tensor) = ad.view(shape)
        let dfTensorRev(a) = ViewT(a, a.shape)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.view(shape:int) = t.view([|shape|])

    member a.viewAs(b:Tensor) = a.view(b.shape)

    member a.flatten(?startDim:int, ?endDim:int) =
        if a.dim < 2 then 
            a
        else
            let startDim = defaultArg startDim 0
            let endDim = defaultArg endDim (a.dim - 1)
            Shape.checkCanFlatten a.shape startDim endDim
            a.view(a.shape |> Shape.flatten startDim endDim)

    member a.sign() =
        let fRaw(a:RawTensor) = a.SignT()
        let fTensor(a:Tensor) = a.sign()
        let dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let dfTensorRev(a) = SignT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    // static member Sign(a:Tensor) = a.sign() // not supported becaose FSharp.Core sign operator returns int

    member a.floor() =
        let fRaw(a:RawTensor) = a.FloorT()
        let fTensor(a:Tensor) = a.floor()
        let dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let dfTensorRev(a) = FloorT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Floor(a:Tensor) = a.floor() // needed for FSharp.Core floor operator overload

    member a.ceil() =
        let fRaw(a:RawTensor) = a.CeilT()
        let fTensor(a:Tensor) = a.ceil()
        let dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let dfTensorRev(a) = CeilT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Ceiling(a:Tensor) = a.ceil() // needed for FSharp.Core ceil operator overload

    member a.round() =
        let fRaw(a:RawTensor) = a.RoundT()
        let fTensor(a:Tensor) = a.round()
        let dfTensorFwd(cp:Tensor,ap,ad) = cp.zerosLike()
        let dfTensorRev(a) = RoundT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Round(a:Tensor) = a.round() // needed for FSharp.Core round operator overload

    member a.abs() =
        let fRaw(a:RawTensor) = a.AbsT()
        let fTensor(a:Tensor) = a.abs()
        let dfTensorFwd(cp,ap:Tensor,ad) = ad * ap.sign()
        let dfTensorRev(a) = AbsT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Abs(a:Tensor) = a.abs() // needed for FSharp.Core abs operator overload

    member a.relu() =
        let fRaw(a:RawTensor) = a.ReluT()
        let fTensor(a:Tensor) = a.relu()
        let dfTensorFwd(cp,ap:Tensor,ad:Tensor) = let sap = ap.sign() in ad * sap.abs() * (sap + 1.) / 2.
        let dfTensorRev(a) = ReluT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.leakyRelu(?negativeSlope:float) =
        let negativeSlope = defaultArg negativeSlope 0.01
        let zeros = a.zerosLike() in zeros.max(a) + negativeSlope * zeros.min(a)

    member a.sigmoid() =
        let fRaw(a:RawTensor) = a.SigmoidT()
        let fTensor(a:Tensor) = a.sigmoid()
        let dfTensorFwd(cp:Tensor,ap,ad) = ad * cp * (1. - cp)
        let dfTensorRev(a) = SigmoidT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.exp() =
        let fRaw(a:RawTensor) = a.ExpT()
        let fTensor(a:Tensor) = a.exp()
        let dfTensorFwd(cp,ap,ad) = ad * cp
        let dfTensorRev(a) = ExpT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Exp(a:Tensor) = a.exp() // needed for FSharp.Core exp operator overload

    member a.log() =
        let fRaw(a:RawTensor) = a.LogT()
        let fTensor(a:Tensor) = a.log()
        let dfTensorFwd(cp,ap,ad) = ad / ap
        let dfTensorRev(a) = LogT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Log(a:Tensor) = a.log() // needed for FSharp.Core log operator overload

    member a.softplus() =
        let fRaw(a:RawTensor) = a.SoftplusT()
        let fTensor(a:Tensor) = a.softplus()
        let dfTensorFwd(cp,ap:Tensor,ad) = ad / (1. + ap.neg().exp())
        let dfTensorRev(a) = SoftplusT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.log10() =
        let fRaw(a:RawTensor) = a.Log10T()
        let fTensor(a:Tensor) = a.log10()
        let dfTensorFwd(cp,ap:Tensor,ad) = ad / (ap * log10Val)
        let dfTensorRev(a) = Log10T(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Log10(a:Tensor) = a.log10() // needed for FSharp.Core log10 operator overload

    member a.sqrt() =
        let fRaw(a:RawTensor) = a.SqrtT()
        let fTensor(a:Tensor) = a.sqrt()
        let dfTensorFwd(cp:Tensor,ap,ad) = ad / (2. * cp)
        let dfTensorRev(a) = SqrtT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sqrt(a:Tensor) = a.sqrt() // needed for FSharp.Core sqrt operator overload

    member a.sin() =
        let fRaw(a:RawTensor) = a.SinT()
        let fTensor(a:Tensor) = a.sin()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.cos()
        let dfTensorRev(a) = SinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sin(a:Tensor) = a.sin() // needed for FSharp.Core sin operator overload

    member a.cos() =
        let fRaw(a:RawTensor) = a.CosT()
        let fTensor(a:Tensor) = a.cos()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = -ad * ap.sin()
        let dfTensorRev(a) = CosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Cos(a:Tensor) = a.cos() // needed for FSharp.Core cos operator overload

    member a.tan() =
        let fRaw(a:RawTensor) = a.TanT()
        let fTensor(a:Tensor) = a.tan()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = let cosap = ap.cos() in ad / (cosap * cosap)
        let dfTensorRev(a) = TanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Tan(a:Tensor) = a.tan() // needed for FSharp.Core tan operator overload

    member a.sinh() =
        let fRaw(a:RawTensor) = a.SinhT()
        let fTensor(a:Tensor) = a.sinh()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.cosh()
        let dfTensorRev(a) = SinhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Sinh(a:Tensor) = a.sinh() // needed for FSharp.Core sinh operator overload

    member a.cosh() =
        let fRaw(a:RawTensor) = a.CoshT()
        let fTensor(a:Tensor) = a.cosh()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad * ap.sinh()
        let dfTensorRev(a) = CoshT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Cosh(a:Tensor) = a.cosh() // needed for FSharp.Core cosh operator overload

    member a.tanh() =
        let fRaw(a:RawTensor) = a.TanhT()
        let fTensor(a:Tensor) = a.tanh()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = let coshap = ap.cosh() in ad / (coshap * coshap)
        let dfTensorRev(a) = TanhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Tanh(a:Tensor) = a.tanh() // needed for FSharp.Core tanh operator overload

    member a.asin() =
        let fRaw(a:RawTensor) = a.AsinT()
        let fTensor(a:Tensor) = a.asin()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / (1. - ap*ap).sqrt()
        let dfTensorRev(a) = AsinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Asin(a:Tensor) = a.asin() // needed for FSharp.Core asin operator overload

    member a.acos() =
        let fRaw(a:RawTensor) = a.AcosT()
        let fTensor(a:Tensor) = a.acos()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = -ad / (1. - ap*ap).sqrt()
        let dfTensorRev(a) = AcosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Acos(a:Tensor) = a.acos() // needed for FSharp.Core acos operator overload

    member a.atan() =
        let fRaw(a:RawTensor) = a.AtanT()
        let fTensor(a:Tensor) = a.atan()
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad) = ad / (1. + ap*ap)
        let dfTensorRev(a) = AtanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    static member Atan(a:Tensor) = a.atan() // needed for FSharp.Core atan operator overload

    member a.addSlice(location:seq<int>, b:Tensor) =
        let location = location |> Seq.toArray
        Shape.checkCanAddSlice a.shape location b.shape
        let fRaw(a:RawTensor,b) = a.AddTTSlice(location, b)
        let fTensor(a:Tensor,b) = a.addSlice(location, b)
        let dfTensorFwdTT(cp,ap,ad:Tensor,bp:Tensor,bd:Tensor) = ad.addSlice(location, bd)
        let dfTensorFwdTC(cp,ap,ad) = ad
        let dfTensorFwdCT(cp:Tensor,bp,bd) = cp.zerosLike().addSlice(location, bd)
        let dfTensorRevTT(a,b) = AddTTSlice(a,location,b)
        let dfTensorRevTC(a,b) = AddTTConstSlice(a)
        let dfTensorRevCT(a,b) = AddTConstTSlice(location,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    member a.softmax(dim:int) =
        if dim < 0 || dim >= a.dim then failwithf "Expecting 0 <= dim (%A) < a.dim (%A)" dim a.dim
        let e = (a - a.max().noDiff()).exp()
        let esum = e.sum(dim, keepDim=true).repeat(dim, a.shape.[dim])
        e / esum

    member a.logsoftmax(dim:int) =
        if dim < 0 || dim >= a.dim then failwithf "Expecting 0 <= dim (%A) < a.dim (%A)" dim a.dim
        a - a.logsumexp(dim, keepDim=true)

    member a.logsumexp(dim:int, ?keepDim:bool) =
        if dim < 0 || dim >= a.dim then failwithf "Expecting 0 <= dim (%A) < a.dim (%A)" dim a.dim
        let keepDim = defaultArg keepDim false
        let amax = a.max().noDiff()
        let e = (a - amax).exp()
        let res = amax + e.sum(dim).add(System.Single.Epsilon).log()
        if keepDim then res.unsqueeze(dim) else res

    member input.mseLoss(target:Tensor, ?reduction:string) = 
        if input.shape <> target.shape then failwithf "Expecting input.shape (%A) and target.shape (%A) to be the same" input.shape target.shape
        let reduction = defaultArg reduction "mean"
        if not (reduction = "none" || reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (none, mean, sum)" reduction
        let z = input - target
        let l = z * z
        if reduction = "none" then
            l
        elif reduction = "mean" then
            l.mean()
        else // reduction = "sum"
            l.sum()

    member input.crossEntropyLoss(target:Tensor, ?weight:Tensor, ?reduction:string) =
        input.logsoftmax(dim=1).nllLoss(target, ?weight=weight, ?reduction=reduction)

    member input.nllLoss(target:Tensor, ?weight:Tensor, ?reduction:string) =
        let n, classes, d = 
            if input.dim < 2 
                then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
            elif input.dim = 2 then
                let n, c = input.shape.[0], input.shape.[1]
                if target.shape <> [|n|] then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                n, c, [||]
            else
                let n, c, d = input.shape.[0], input.shape.[1], input.shape.[2..]
                if target.shape.[0] <> n then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                if d <> target.shape.[1..] then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                n, c, d
        let mutable weightSpecified = false
        let mutable ww = input.zeroLike()
        match weight with
        | Some w -> ww <- w; weightSpecified <- true
        | None -> ww <- input.onesLike([classes]); weightSpecified <- false
        let weight = ww
        let reduction = defaultArg reduction "mean"
        if not (reduction = "none" || reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (none, mean, sum)" reduction
        if input.dim = 2 then
            let mutable wacc = input.zeroLike()
            let l = Array.init n (fun i -> 
                                    let target = int target.[i]
                                    let w = weight.[target]
                                    wacc <- wacc + w
                                    -w*input.[i, target]) |> Tensor.stack
            if reduction = "none" then
                l
            elif reduction = "mean" then
                if weightSpecified then l.sum()/wacc else l.mean()
            else // reduction = "sum"
                l.sum()
        else
            let mutable wacc = input.zeroLike()
            let l = Array.init n (fun i ->
                                    let aa = input.[i].view([classes; -1])
                                    let bb = target.[i].view(-1)
                                    let l = Array.init bb.nelement (fun j ->
                                                                    let target = int bb.[j]
                                                                    let w = weight.[target]
                                                                    wacc <- wacc + w
                                                                    -w*aa.[target, j]) |> Tensor.stack
                                    l.view(d)) |> Tensor.stack
            if reduction = "none" then
                l
            elif reduction = "mean" then
                if weightSpecified then l.sum()/wacc else l.mean()
            else // reduction = "sum"
                l.sum()

    member a.pad(paddings:seq<int>) =
        let paddings = paddings |> Array.ofSeq
        Shape.checkCanPad a.shape paddings
        if paddings |> Array.sum = 0 then
            a
        else
            let shape = Array.copy a.shape
            for i in 0..shape.Length-1 do
                shape.[i] <- shape.[i] + paddings.[i] * 2
            let ret = a.zerosLike(shape)
            ret.addSlice(paddings, a)

    member a.maxpool1di(kernelSize:int, ?stride:int, ?padding:int) =
        let stride = defaultArg stride kernelSize
        let padding = defaultArg padding 0
        Shape.checkCanMaxpool1d a.shape kernelSize stride padding  |> ignore
        match a with
        | Tensor(ap)           -> let result, indices = ap.MaxPool1D(kernelSize, stride, padding) in Tensor(result), Tensor(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool1di(kernelSize, stride, padding) in TensorF(result, ad.gather(dim=2, indices=indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool1di(kernelSize, stride, padding) in TensorR(result, ref (a.zeroLike()), MaxPool1DT(a, indices, kernelSize), ref 0u, at), indices

    member a.maxpool1d(kernelSize:int, ?stride:int, ?padding:int) = a.maxpool1di(kernelSize, ?stride=stride, ?padding=padding) |> fst

    member a.maxunpool1d(indices:Tensor, kernelSize:int, ?stride:int, ?padding:int, ?outputSize:seq<int>) =
        let stride = defaultArg stride kernelSize
        let padding = defaultArg padding 0
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 3 then failwithf "Expecting outputSize to be 3-dimensional" else o
            | None -> 
                let inputSize = a.shape.[2]
                [|indices.shape.[0]; indices.shape.[1]; ((inputSize-1) * stride - 2*padding + kernelSize)|]
        Shape.checkCanMaxunpool1d a.shape indices.dtype indices.shape outputSize |> ignore
        let fRaw(a:RawTensor) = a.MaxUnpool1D(indices.primalRaw, outputSize)
        let fTensor(a:Tensor) = a.maxunpool1d(indices, kernelSize, stride=stride, padding=padding, outputSize=outputSize)
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad:Tensor) = ad.maxunpool1d(indices, kernelSize, stride=stride, padding=padding, outputSize=outputSize)
        let dfTensorRev(a) = MaxUnpool1DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.maxpool2di(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 2 then failwithf "Expecting kernelSizes to be 2-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"
        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 2 then failwithf "Expecting strides to be 2-dimensional" else s
            | _ -> kernelSizes
        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting paddings to be 2-dimensional" else p
            | _ -> [|0; 0|]
        Shape.checkCanMaxpool2d a.shape kernelSizes strides paddings  |> ignore
        match a with
        | Tensor(ap)           -> let result, indices = ap.MaxPool2D(kernelSizes, strides, paddings) in Tensor(result), Tensor(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool2di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorF(result, ad.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool2di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorR(result, ref (a.zeroLike()), MaxPool2DT(a, indices, kernelSizes), ref 0u, at), indices

    member a.maxpool2d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool2di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

    member a.maxunpool2d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 2 then failwithf "Expecting kernelSizes to be 2-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"
        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 2 then failwithf "Expecting strides to be 2-dimensional" else s
            | _ -> kernelSizes
        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting paddings to be 2-dimensional" else p
            | _ -> [|0; 0|]
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 4 then failwithf "Expecting outputSize to be 4-dimensional" else o
            | None -> 
                let inputHeight = a.shape.[2]
                let inputWidth = a.shape.[3]
                [|indices.shape.[0]; indices.shape.[1]; ((inputHeight-1) * strides.[0] - 2*paddings.[0] + kernelSizes.[0]); ((inputWidth-1) * strides.[1] - 2*paddings.[1] + kernelSizes.[1])|]
        Shape.checkCanMaxunpool2d a.shape indices.dtype indices.shape outputSize |> ignore
        let fRaw(a:RawTensor) = a.MaxUnpool2D(indices.primalRaw, outputSize)
        let fTensor(a:Tensor) = a.maxunpool2d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad:Tensor) = ad.maxunpool2d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let dfTensorRev(a) = MaxUnpool2DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.maxpool3di(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 3 then failwithf "Expecting kernelSizes to be 3-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"
        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 3 then failwithf "Expecting strides to be 3-dimensional" else s
            | _ -> kernelSizes
        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting paddings to be 3-dimensional" else p
            | _ -> [|0; 0; 0|]
        Shape.checkCanMaxpool3d a.shape kernelSizes strides paddings |> ignore
        match a with
        | Tensor(ap)           -> let result, indices = ap.MaxPool3D(kernelSizes, strides, paddings) in Tensor(result), Tensor(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool3di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorF(result, ad.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool3di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorR(result, ref (a.zeroLike()), MaxPool3DT(a, indices, kernelSizes), ref 0u, at), indices

    member a.maxpool3d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool3di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

    member a.maxunpool3d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 3 then failwithf "Expecting kernelSizes to be 3-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"
        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 3 then failwithf "Expecting strides to be 3-dimensional" else s
            | _ -> kernelSizes
        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting paddings to be 3-dimensional" else p
            | _ -> [|0; 0; 0|]
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 5 then failwithf "Expecting outputSize to be 5-dimensional" else o
            | None -> 
                let inputDepth = a.shape.[2]
                let inputHeight = a.shape.[3]
                let inputWidth = a.shape.[4]
                [|indices.shape.[0]; indices.shape.[1]; ((inputDepth-1) * strides.[0] - 2*paddings.[0] + kernelSizes.[0]); ((inputHeight-1) * strides.[1] - 2*paddings.[1] + kernelSizes.[1]); ((inputWidth-1) * strides.[2] - 2*paddings.[2] + kernelSizes.[2])|]
        Shape.checkCanMaxunpool3d a.shape indices.dtype indices.shape outputSize |> ignore
        let fRaw(a:RawTensor) = a.MaxUnpool3D(indices.primalRaw, outputSize)
        let fTensor(a:Tensor) = a.maxunpool3d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let dfTensorFwd(cp:Tensor,ap:Tensor,ad:Tensor) = ad.maxunpool3d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let dfTensorRev(a) = MaxUnpool3DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)

    member a.conv1d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int) =
        // a: input, b: filter
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let dilation = defaultArg dilation 1
        Shape.checkCanConv1d a.dtype b.dtype a.shape b.shape stride padding dilation |> ignore
        let mutable b = b
        if dilation > 1 then
            b <- b.dilate([|1;1;dilation|])
        let fRaw(a:RawTensor,b) = a.Conv1D(b, stride, padding)
        let fTensor(a:Tensor,b) = a.conv1d(b, stride, padding)
        let dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor) = ad.conv1d(bp, stride, padding) + ap.conv1d(bd, stride, padding)
        let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.conv1d(b, stride, padding)
        let dfTensorFwdCT(cp,bp,bd) = a.conv1d(bd, stride, padding)
        let dfTensorRevTT(a,b) = Conv1DTT(a,b, stride, padding)
        let dfTensorRevTC(a,b) = Conv1DTTConst(a,b, stride, padding)
        let dfTensorRevCT(a,b) = Conv1DTConstT(a,b, stride, padding)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    // a: input, NxCxI (batchSize x inputChannels x inputLength)
    // b: filters, KxCxF (outputChannels x inputChannels x kernelLength)
    // t: output, NxKxL (batchSize x outputChannels x outputLength)
    member internal t.conv1dReverseDiff(a: Tensor, b:Tensor, aConst:bool, bConst:bool, stride:int, padding:int) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = t.shape.[0]
        let outputChannels = t.shape.[1]
        // let outputLength = t.shape.[2]
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
                    let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding; padding + inputLength - 1; 1]]
                    c <- c.GetSlice(cBounds)
                    c <- c.view([|batchSize; inputChannels; inputLength|])
                aderivative <- aderivative + c
        if not bConst then
            // propagate to b
            bderivative <- b.zerosLike()
            for n=0 to batchSize-1 do
                let aa = a.[n].view([|inputChannels; 1; inputLength|]) // treat size-one batch of a c-channel image as a size-c batch of one-channel images
                let d = tderivative.[n]
                for k=0 to outputChannels-1 do
                    let dd = d.[k].view([|1; 1; tderivative.shape.[2]|])
                    let mutable c = aa.conv1d(dd, padding=padding)
                    c <- c.view([|1; inputChannels; c.shape.[2]|])
                    let cBounds = array2D [[0;0;1]; [0;inputChannels-1;1]; [0;kernelLength-1;1]]
                    c <- c.GetSlice(cBounds)                 
                    c <- c.view([|1; inputChannels; kernelLength|])
                    bderivative <- bderivative.addSlice([|k; 0; 0|], c)
        aderivative, bderivative

    member a.conv2d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>) =
        let strides = 
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 2 then failwithf "Expecting strides to be 2-dimensional" else s
            | _ -> [|1; 1|]
        let paddings = 
            match padding, paddings with
            | Some _ , Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting paddings to be 2-dimensional" else p
            | _ -> [|0; 0|]
        let dilations = 
            match dilation, dilations with
            | Some _ , Some _ -> failwithf "Expecting only one of dilation, dilations"
            | Some d, None -> [|d; d|]
            | None, Some d -> let d = d |> Array.ofSeq in if d.Length <> 2 then failwithf "Expecting dilations to be 2-dimensional" else d
            | _ -> [|1; 1|]
        Shape.checkCanConv2d a.dtype b.dtype a.shape b.shape strides paddings dilations |> ignore
        let mutable b = b
        if dilations.[0] > 1 || dilations.[1] > 1 then
            b <- b.dilate([|1; 1; dilations.[0]; dilations.[1]|])
        let fRaw(a:RawTensor,b) = a.Conv2D(b, strides, paddings)
        let fTensor(a:Tensor,b) = a.conv2d(b, strides=strides, paddings=paddings)
        let dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp,bd) = ad.conv2d(bp, strides=strides, paddings=paddings) + ap.conv2d(bd, strides=strides, paddings=paddings)
        let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.conv2d(b, strides=strides, paddings=paddings)
        let dfTensorFwdCT(cp,bp,bd) = a.conv2d(bd, strides=strides, paddings=paddings)
        let dfTensorRevTT(a,b) = Conv2DTT(a,b, strides, paddings)
        let dfTensorRevTC(a,b) = Conv2DTTConst(a,b, strides, paddings)
        let dfTensorRevCT(a,b) = Conv2DTConstT(a,b, strides, paddings)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    // a: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
    // b: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
    // t: output, NxKxLxM (batchSize x outputChannels x outputHeight x outputWidth)
    member internal t.conv2dReverseDiff(a: Tensor, b:Tensor, aConst:bool, bConst:bool, strides:int[], paddings:int[]) =
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
        if strides.[0] > 1 || strides.[1] > 1 then
            tderivative <- tderivative.dilate([|1;1;strides.[0];strides.[1]|])
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
                let mutable c : Tensor = d.conv2d(b, paddings=[|kernelHeight-1; kernelWidth-1|])
                if paddings.[0] > 0 || paddings.[1] > 0 then
                    let cBounds = array2D [[0; batchSize-1; 1]; 
                                           [0; inputChannels-1; 1]; 
                                           [paddings.[0]; paddings.[0] + inputHeight - 1; 1]; 
                                           [paddings.[1]; paddings.[1] + inputWidth - 1; 1]]
                    c <- c.GetSlice(cBounds)
                    c <- c.view([|batchSize; inputChannels; inputHeight; inputWidth|])
                aderivative <- aderivative  + c
        if not bConst then
            // propagate to b
            bderivative <- b.zerosLike()
            for n=0 to batchSize-1 do
                let aa = a.[n].view([|inputChannels; 1; inputHeight; inputWidth|]) // treat size-one batch of a c-channel image as a size-c batch of one-channel images
                let d = tderivative.[n]
                for k=0 to outputChannels-1 do
                    let dd = d.[k].view([|1; 1; tderivative.shape.[2]; tderivative.shape.[3]|])
                    let mutable c = aa.conv2d(dd, paddings=paddings)
                    // c <- c.view([|1; inputChannels; kernelHeight; kernelWidth|])
                    c <- c.view([|1; inputChannels; c.shape.[2]; c.shape.[3]|])
                    let cBounds = array2D [[0;0;1]; [0;inputChannels-1;1]; [0;kernelHeight-1;1]; [0;kernelWidth-1;1]]
                    c <- c.GetSlice(cBounds)                 
                    c <- c.view([|1; inputChannels; kernelHeight; kernelWidth|])
                    bderivative <- bderivative.addSlice([|k; 0; 0; 0|], c)
        aderivative, bderivative

    member a.conv3d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>) =
        let strides = 
            match stride, strides with
            | Some _ , Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 3 then failwithf "Expecting strides to be 3-dimensional" else s
            | _ -> [|1; 1; 1|]
        let paddings = 
            match padding, paddings with
            | Some _ , Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting paddings to be 3-dimensional" else p
            | _ -> [|0; 0; 0|]
        let dilations = 
            match dilation, dilations with
            | Some _ , Some _ -> failwithf "Expecting only one of dilation, dilations"
            | Some d, None -> [|d; d; d|]
            | None, Some d -> let d = d |> Array.ofSeq in if d.Length <> 3 then failwithf "Expecting dilations to be 3-dimensional" else d
            | _ -> [|1; 1; 1|]
        Shape.checkCanConv3d a.dtype b.dtype a.shape b.shape strides paddings dilations |> ignore
        let mutable b = b
        if dilations.[0] > 1 || dilations.[1] > 1 || dilations.[2] > 1 then
            b <- b.dilate([|1; 1; dilations.[0]; dilations.[1]; dilations.[2]|])
        let fRaw(a:RawTensor,b) = a.Conv3D(b, strides, paddings)
        let fTensor(a:Tensor,b) = a.conv3d(b, strides=strides, paddings=paddings)
        let dfTensorFwdTT(cp,ap:Tensor,ad:Tensor,bp,bd) = ad.conv3d(bp, strides=strides, paddings=paddings) + ap.conv3d(bd, strides=strides, paddings=paddings)
        let dfTensorFwdTC(cp,ap,ad:Tensor) = ad.conv3d(b, strides=strides, paddings=paddings)
        let dfTensorFwdCT(cp,bp,bd) = a.conv3d(bd, strides=strides, paddings=paddings)
        let dfTensorRevTT(a,b) = Conv3DTT(a,b, strides, paddings)
        let dfTensorRevTC(a,b) = Conv3DTTConst(a,b, strides, paddings)
        let dfTensorRevCT(a,b) = Conv3DTConstT(a,b, strides, paddings)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)

    // a: input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth)
    // b: filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth)
    // t: output, NxKxLxMxN (batchSize x outputChannels x outputDepth x outputHeight x outputWidth)
    member internal t.conv3dReverseDiff(a: Tensor, b:Tensor, aConst:bool, bConst:bool, strides:int[], paddings:int[]) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = t.shape.[0]
        let outputChannels = t.shape.[1]
        // let outputDepth = t.shape.[2]
        // let outputHeight = t.shape.[3]
        // let outputWidth = t.shape.[4]
        let inputChannels = a.shape.[1]
        let inputDepth = a.shape.[2]
        let inputHeight = a.shape.[3]
        let inputWidth = a.shape.[4]
        let kernelDepth = b.shape.[2]
        let kernelHeight = b.shape.[3]
        let kernelWidth = b.shape.[4]
        let mutable tderivative = t.derivative
        if strides.[0] > 1 || strides.[1] > 1 || strides.[2] > 1 then
            tderivative <- tderivative.dilate([|1;1;strides.[0];strides.[1];strides.[2]|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            aderivative <- a.zerosLike()
            let bFlipped = b.flip([|2;3;4|])
            for k=0 to outputChannels-1 do
                let b = bFlipped.[k].view([|inputChannels; 1; kernelDepth; kernelHeight; kernelWidth|])
                let dBounds = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.shape.[2]-1; 1]; [0; tderivative.shape.[3]-1; 1]; [0; tderivative.shape.[4]-1; 1]]
                let d = tderivative.GetSlice(dBounds).view([|batchSize; 1; tderivative.shape.[2]; tderivative.shape.[3]; tderivative.shape.[4]|])
                let mutable c : Tensor = d.conv3d(b, paddings=[|kernelDepth-1; kernelHeight-1; kernelWidth-1|])
                if paddings.[0] > 0 || paddings.[1] > 0 || paddings.[2] > 0 then
                    let cBounds = array2D [[0; batchSize-1; 1]; 
                                           [0; inputChannels-1; 1]; 
                                           [paddings.[0]; paddings.[0] + inputDepth - 1; 1]; 
                                           [paddings.[1]; paddings.[1] + inputHeight - 1; 1];
                                           [paddings.[2]; paddings.[2] + inputWidth - 1; 1]]
                    c <- c.GetSlice(cBounds)
                    c <- c.view([|batchSize; inputChannels; inputDepth; inputHeight; inputWidth|])
                aderivative <- aderivative  + c
        if not bConst then
            // propagate to b
            bderivative <- b.zerosLike()
            for n=0 to batchSize-1 do
                let aa = a.[n].view([|inputChannels; 1; inputDepth; inputHeight; inputWidth|]) // treat size-one batch of a c-channel image as a size-c batch of one-channel images
                let d = tderivative.[n]
                for k=0 to outputChannels-1 do
                    let dd = d.[k].view([|1; 1; tderivative.shape.[2]; tderivative.shape.[3]; tderivative.shape.[4]|])
                    let mutable c = aa.conv3d(dd, paddings=paddings)
                    // c <- c.view([|1; inputChannels; kernelHeight; kernelWidth|])
                    c <- c.view([|1; inputChannels; c.shape.[2]; c.shape.[3]; c.shape.[4]|])
                    let cBounds = array2D [[0;0;1]; [0;inputChannels-1;1]; [0;kernelDepth-1;1]; [0;kernelHeight-1;1]; [0;kernelWidth-1;1]]
                    c <- c.GetSlice(cBounds)
                    c <- c.view([|1; inputChannels; kernelDepth; kernelHeight; kernelWidth|])
                    bderivative <- bderivative.addSlice([|k; 0; 0; 0; 0|], c)
        aderivative, bderivative

    member t.reverse(?value:Tensor, ?zeroDerivatives:bool) =
        let value = defaultArg value (t.onesLike())
        let zeroDerivatives = defaultArg zeroDerivatives true
        if value.shape <> t.shape then failwithf "Expecting value.shape (%A) and t.shape (%A) to be the same" value.shape t.shape
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
                        | MaxPool1DT(a,_,_) -> reset (a::tt)
                        | MaxPool2DT(a,_,_) -> reset (a::tt)
                        | MaxPool3DT(a,_,_) -> reset (a::tt)
                        | MaxUnpool1DT(a,_) -> reset (a::tt)
                        | MaxUnpool2DT(a,_) -> reset (a::tt)
                        | MaxUnpool3DT(a,_) -> reset (a::tt)
                        | Conv1DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv1DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv1DTConstT(_,b,_,_) -> reset (b::tt)
                        | Conv2DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv2DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv2DTConstT(_,b,_,_) -> reset (b::tt)
                        | Conv3DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv3DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv3DTConstT(_,b,_,_) -> reset (b::tt)
                        | NegT(a) -> reset (a::tt)
                        | SumT(a) -> reset (a::tt)
                        | SumT2Dim0(a) -> reset (a::tt)
                        | ExpandT(a) -> reset (a::tt)
                        | StackTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | UnstackT(a,_,_) -> reset (a::tt)
                        | CatTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | SplitT(a,_,_,_) -> reset (a::tt)
                        | GatherT(a,_,_) -> reset (a::tt)
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
                        | SoftplusT(a) -> reset (a::tt)
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
                    // if t.derivative.hasnan() || t.derivative.hasinf() then failwithf "t.derivative has nan, inf, or -inf\n%A\n%A" t.derivative t.derivative.shape
                    // if v.hasnan() || v.hasinf() then failwithf "v has nan, inf, or -inf\n%A\n%A\n%s" v v.shape (snd (t.parents()))
                    t.derivative <- t.derivative + v
                    t.fanout <- t.fanout - 1u
                    if t.fanout = 0u then
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
                        | MaxPool1DT(a, indices, kernelSize) -> push ((t.derivative.maxunpool1d(indices, kernelSize=kernelSize, outputSize=a.shape), a) :: tt)
                        | MaxPool2DT(a, indices, kernelSizes) -> push ((t.derivative.maxunpool2d(indices, kernelSizes=kernelSizes, outputSize=a.shape), a) :: tt)
                        | MaxPool3DT(a, indices, kernelSizes) -> push ((t.derivative.maxunpool3d(indices, kernelSizes=kernelSizes, outputSize=a.shape), a) :: tt)
                        | MaxUnpool1DT(a, indices) -> push ((t.derivative.gather(dim=2, indices=indices), a) :: tt)
                        | MaxUnpool2DT(a, indices) -> push ((t.derivative.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(a), a) :: tt)
                        | MaxUnpool3DT(a, indices) -> push ((t.derivative.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(a), a) :: tt)
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
                        | Conv3DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = t.conv3dReverseDiff(a, b, false, false, stride, padding)
                            push ((aderivative, a) :: (bderivative, b) :: tt)
                        | Conv3DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = t.conv3dReverseDiff(a, b, false, true, stride, padding)
                            push ((aderivative, a) :: tt)
                        | Conv3DTConstT(a,b,stride,padding) ->
                            let _, bderivative = t.conv3dReverseDiff(a, b, true, false, stride, padding)
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
                        | GatherT(a,dim,indices) -> 
                            // TODO: The following is a minimal correct implementation. Faster and more memory efficient implementations should be possible.
                            let tflat = t.derivative.flatten()
                            let iflat = indices.flatten()
                            if a.derivative.dim = 0 then a.derivative <- a.zerosLike() + a.derivative
                            for i=0 to tflat.nelement-1 do
                                let mutable t = tflat.[i]
                                for k=0 to a.dim-1 do
                                    t <- t.unsqueeze(0)
                                let j = iflat.[i].toScalar() :?> int
                                let loc = flatIndexToIndex a.shape i
                                loc.[dim] <- j
                                a.derivative <- a.derivative.addSlice(loc, t)
                            push ((a.zeroLike(), a) :: tt)
                        | TransposeT2(a) -> push ((t.derivative.transpose(), a) :: tt)
                        | SqueezeT(a) -> push ((t.derivative.viewAs(a), a) :: tt)
                        | UnsqueezeT(a) -> push ((t.derivative.viewAs(a), a) :: tt)
                        | FlipT(a, dims) -> push ((t.derivative.flip(dims), a) :: tt)
                        | DilateT(a, dilations) -> push ((t.derivative.undilate(dilations), a) :: tt)
                        | UndilateT(a, dilations) -> push ((t.derivative.dilate(dilations), a) :: tt)
                        | ViewT(a,aShape) -> push (((t.derivative.view(aShape)), a) :: tt)
                        | SliceT(a,bounds) -> 
                            // TODO: a.zerosLike() below is to handle non-scalar TensorRs with a scalar derivative Tensor(0.) (representing the initialization before accumulation). This is correct but can be changed to eliminate the extra op.
                            if a.derivative.dim = 0 then a.derivative <- a.zerosLike() + a.derivative
                            a.derivative <- a.derivative.addSlice(boundsToLocation bounds, t.derivative.view(boundsToShape bounds))
                            push ((a.zeroLike(), a) :: tt)
                        | AddTTSlice(a,location,b) -> push ((t.derivative, a) :: (t.derivative.GetSlice(Shape.locationToBounds b.shape location), b):: tt)
                        | AddTTConstSlice(a) -> push ((t.derivative, a) :: tt)
                        | AddTConstTSlice(location, b) -> push ((t.derivative.GetSlice(Shape.locationToBounds b.shape location), b):: tt)
                        | SignT(a) -> push ((a.zerosLike(), a) :: tt)
                        | FloorT(a) -> push ((a.zerosLike(), a) :: tt)
                        | CeilT(a) -> push ((a.zerosLike(), a) :: tt)
                        | RoundT(a) -> push ((a.zerosLike(), a) :: tt)
                        | AbsT(a) -> push ((t.derivative * a.primal.sign(), a) :: tt)
                        | ReluT(a) -> let sap = a.primal.sign() in push ((t.derivative * (sap.abs()) * (sap + 1.) / 2., a) :: tt)
                        | SoftplusT(a) -> push ((t.derivative / (1. + a.primal.neg().exp()), a) :: tt)
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
    | AddTT of Tensor * Tensor// derivative test implemented
    | AddTTConst of Tensor // derivative test implemented
    | AddTT0 of Tensor * Tensor // derivative test implemented
    | AddTT0Const of Tensor // derivative test implemented
    | AddTConstT0 of Tensor // derivative test implemented
    | AddT2T1 of Tensor * Tensor // derivative test implemented
    | AddT2T1Const of Tensor // derivative test implemented
    | AddT2ConstT1 of Tensor // derivative test implemented
    
    | SubTT of Tensor * Tensor // derivative test implemented
    | SubTTConst of Tensor // derivative test implemented
    | SubTConstT of Tensor // derivative test implemented
    | SubT0T of Tensor * Tensor // derivative test implemented
    | SubT0TConst of Tensor // derivative test implemented
    | SubT0ConstT of Tensor // derivative test implemented
    | SubTT0 of Tensor * Tensor // derivative test implemented
    | SubTT0Const of Tensor // derivative test implemented
    | SubTConstT0 of Tensor // derivative test implemented

    | MulTT of Tensor * Tensor // derivative test implemented
    | MulTTConst of Tensor * Tensor // derivative test implemented
    | MulTT0 of Tensor * Tensor // derivative test implemented
    | MulTT0Const of Tensor * Tensor // derivative test implemented
    | MulTConstT0 of Tensor * Tensor // derivative test implemented

    | DivTT of Tensor * Tensor // derivative test implemented
    | DivTTConst of Tensor * Tensor // derivative test implemented
    | DivTConstT of Tensor * Tensor // derivative test implemented
    | DivT0T of Tensor * Tensor // derivative test implemented
    | DivT0TConst of Tensor * Tensor // derivative test implemented
    | DivT0ConstT of Tensor * Tensor // derivative test implemented
    | DivTT0 of Tensor * Tensor // derivative test implemented
    | DivTT0Const of Tensor * Tensor // derivative test implemented
    | DivTConstT0 of Tensor * Tensor // derivative test implemented

    | PowTT of Tensor * Tensor // derivative test implemented
    | PowTTConst of Tensor * Tensor // derivative test implemented
    | PowTConstT of Tensor * Tensor // derivative test implemented
    | PowT0T of Tensor * Tensor // derivative test implemented
    | PowT0TConst of Tensor * Tensor // derivative test implemented
    | PowT0ConstT of Tensor * Tensor // derivative test implemented
    | PowTT0 of Tensor * Tensor // derivative test implemented
    | PowTT0Const of Tensor * Tensor // derivative test implemented
    | PowTConstT0 of Tensor * Tensor // derivative test implemented

    | MatMulT2T2 of Tensor * Tensor // derivative test implemented
    | MatMulT2T2Const of Tensor * Tensor // derivative test implemented
    | MatMulT2ConstT2 of Tensor * Tensor // derivative test implemented

    | MaxPool1DT of Tensor * Tensor * int
    | MaxUnpool1DT of Tensor * Tensor

    | MaxPool2DT of Tensor * Tensor * int[]
    | MaxUnpool2DT of Tensor * Tensor

    | MaxPool3DT of Tensor * Tensor * int[]
    | MaxUnpool3DT of Tensor * Tensor

    | Conv1DTT of Tensor * Tensor * int * int
    | Conv1DTTConst of Tensor * Tensor * int * int
    | Conv1DTConstT of Tensor * Tensor * int * int

    | Conv2DTT of Tensor * Tensor * int[] * int[]
    | Conv2DTTConst of Tensor * Tensor * int[] * int[]
    | Conv2DTConstT of Tensor * Tensor * int[] * int[]

    | Conv3DTT of Tensor * Tensor * int[] * int[]
    | Conv3DTTConst of Tensor * Tensor * int[] * int[]
    | Conv3DTConstT of Tensor * Tensor * int[] * int[]

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
    | GatherT of Tensor * int * Tensor
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
    | SoftplusT of Tensor
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
    [<ExcludeFromCodeCoverage>]
    member t.GetSlice(i0min:int option, i0max:int option) =
        // Dims: 1
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    member t.GetSlice(i0:int) =
        // Dims: 1
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
    [<ExcludeFromCodeCoverage>]
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
