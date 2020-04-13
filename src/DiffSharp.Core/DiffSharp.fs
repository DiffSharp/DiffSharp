namespace DiffSharp
open DiffSharp.Backend
open DiffSharp.Util

// Tensor operations
type DiffSharp =
    static member tensor(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) = 
        let array, shape = value |> flatArrayAndShape<Tensor> // support creation of new Tensor from a structure holding scalar Tensors
        if notNull array then 
            let array = array |> Array.map float32
            let value = arrayND shape (fun ii -> array.[indexToFlatIndex shape ii])
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))
        else
            Tensor(RawTensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend))
    static member zero(?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zero(?dtype=dtype, ?device=device, ?backend=backend))
    static member zeros(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zeros(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))
    static member one(?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.One(?dtype=dtype, ?device=device, ?backend=backend))
    static member ones(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Ones(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))
    static member onehot(length:int, hot:int, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zeros([||], ?dtype=dtype, ?device=device, ?backend=backend)).onehotLike(length, hot)
    static member rand(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Random(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))
    static member randn(shape:seq<int>, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor(RawTensor.RandomNormal(shape|>Seq.toArray, ?dtype=dtype, ?device=device, ?backend=backend))
    static member zerosLike(a:Tensor, ?shape:seq<int>) = a.zerosLike(?shape=shape)
    static member onesLike(a:Tensor, ?shape:seq<int>) = a.onesLike(?shape=shape)
    static member randLike(a:Tensor, ?shape:seq<int>) = a.randLike(?shape=shape)
    static member randnLike(a:Tensor, ?shape:seq<int>) = a.randnLike(?shape=shape)
    static member zeroLike(a:Tensor) = a.zeroLike()
    static member oneLike(a:Tensor) = a.oneLike()
    static member arange(endVal:float, ?startVal:float, ?step:float, ?dtype:DType, ?device:Device, ?backend:Backend) = DiffSharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)
    static member like(a:Tensor, value:obj) = a.like(value)
    static member clone(a:Tensor) = a.clone()
    static member lt(a:Tensor, b:Tensor) = a.lt(b)
    static member gt(a:Tensor, b:Tensor) = a.gt(b)
    static member le(a:Tensor, b:Tensor) = a.le(b)
    static member ge(a:Tensor, b:Tensor) = a.ge(b)
    static member maxIndex(a:Tensor) = a.maxIndex()
    static member minIndex(a:Tensor) = a.minIndex()
    static member max(a:Tensor) = a.max()
    static member min(a:Tensor) = a.min()
    static member max(a:Tensor, b:Tensor) = a.max(b)
    static member min(a:Tensor, b:Tensor) = a.min(b)
    static member expand(a:Tensor, shape:seq<int>) = a.expand(shape)
    static member stack(tensors:seq<Tensor>, ?dim:int) = Tensor.stack(tensors, ?dim=dim)
    static member unstack(a:Tensor, ?dim:int) = a.unstack(?dim=dim)
    static member cat(tensors:seq<Tensor>, ?dim:int) = Tensor.cat(tensors, ?dim=dim)
    static member split(a:Tensor, sizes:seq<int>, ?dim:int) = a.split(sizes, ?dim=dim)
    static member add(a:Tensor, b:Tensor) = a.add(b)
    static member sub(a:Tensor, b:Tensor) = a.sub(b)
    static member mul(a:Tensor, b:Tensor) = a.mul(b)
    static member div(a:Tensor, b:Tensor) = a.div(b)
    static member pow(a:Tensor, b:Tensor) = a.pow(b)
    static member matmul(a:Tensor, b:Tensor) = a.matmul(b)
    static member dot(a:Tensor, b:Tensor) = a.dot(b)
    static member neg(a:Tensor) = a.neg()
    static member sum(a:Tensor) = a.sum()
    static member sum(a:Tensor, dim:int, ?keepDim:bool) = a.sum(dim, ?keepDim=keepDim)
    static member mean(a:Tensor) = a.mean()
    static member mean(a:Tensor, dim:int, ?keepDim:bool) = a.mean(dim, ?keepDim=keepDim)
    static member variance(a:Tensor) = a.variance()
    static member variance(a:Tensor, dim:int, ?keepDim:bool) = a.variance(dim, ?keepDim=keepDim)
    static member stddev(a:Tensor) = a.stddev()
    static member stddev(a:Tensor, dim:int, ?keepDim:bool) = a.stddev(dim, ?keepDim=keepDim)
    static member transpose(a:Tensor) = a.transpose()
    static member squeeze(a:Tensor, ?dim:int) = a.squeeze(?dim=dim)
    static member unsqueeze(a:Tensor, dim:int) = a.unsqueeze(dim)
    static member flip(a:Tensor, dims:seq<int>) = a.flip(dims)
    static member dilate(a:Tensor, dilations:seq<int>) = a.dilate(dilations)
    static member undilate(a:Tensor, dilations:seq<int>) = a.undilate(dilations)
    static member repeat(a:Tensor, dim:int, times:int) = a.repeat(dim, times)
    static member view(a:Tensor, shape:seq<int>) = a.view(shape)
    static member view(a:Tensor, shape:int) = a.view(shape)
    static member viewAs(a:Tensor, b:Tensor) = a.viewAs(b)
    static member sign(a:Tensor) = a.sign()
    static member floor(a:Tensor) = a.floor()
    static member ceil(a:Tensor) = a.ceil()
    static member round(a:Tensor) = a.round()
    static member abs(a:Tensor) = a.abs()
    static member relu(a:Tensor) = a.relu()
    static member leakyRelu(a:Tensor, ?negativeSlope:float) = a.leakyRelu(?negativeSlope=negativeSlope)
    static member sigmoid(a:Tensor) = a.sigmoid()
    static member exp(a:Tensor) = a.exp()
    static member log(a:Tensor) = a.log()
    static member log10(a:Tensor) = a.log10()
    static member sqrt(a:Tensor) = a.sqrt()
    static member sin(a:Tensor) = a.sin()
    static member cos(a:Tensor) = a.cos()
    static member tan(a:Tensor) = a.tan()
    static member sinh(a:Tensor) = a.sinh()
    static member cosh(a:Tensor) = a.cosh()
    static member tanh(a:Tensor) = a.tanh()
    static member asin(a:Tensor) = a.asin()
    static member acos(a:Tensor) = a.acos()
    static member atan(a:Tensor) = a.atan()
    static member softmax(a:Tensor, dim:int) = a.softmax(dim)
    static member mseLoss(a:Tensor, b:Tensor) = a.mseLoss(b)
    static member conv1d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int) = a.conv1d(b, ?stride=stride, ?padding=padding, ?dilation=dilation)
    static member conv2d(a:Tensor, b:Tensor, ?stride:seq<int>, ?padding:seq<int>, ?dilation:seq<int>) = a.conv2d(b, ?stride=stride, ?padding=padding, ?dilation=dilation)
    static member conv2d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int) = a.conv2d(b, ?stride=stride, ?padding=padding, ?dilation=dilation)


// Functional differentiation API
type DiffSharp with
    static member seed(seed) = Random.Seed(seed)
    static member nest() = GlobalNestingLevel.Next() |> ignore
    static member nest(level) = GlobalNestingLevel.Set(level)
    static member nestLevel() = GlobalNestingLevel.Current
    static member nestReset() = GlobalNestingLevel.Reset()
    static member primal (tensor:Tensor) = tensor.primal
    static member derivative (tensor:Tensor) = tensor.derivative
    static member primalDerivative (tensor:Tensor) = tensor.primal, tensor.derivative
    static member forwardDiff (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.forwardDiff(derivative, tag)
    static member reverseDiff (tag:uint32) (tensor:Tensor) = tensor.reverseDiff(tag)
    static member reverseReset (tensor:Tensor) = tensor.reverseReset(true)
    static member reversePush (value:Tensor) (tensor:Tensor) = tensor.reversePush(value)
    static member reverse (value:Tensor) (tensor:Tensor) = tensor.reverse(value)
    static member evalForwardDiff f x v = x |> DiffSharp.forwardDiff (GlobalNestingLevel.Next()) v |> f |> DiffSharp.primalDerivative
    static member evalReverseDiff f x =
        let x = x |> DiffSharp.reverseDiff (GlobalNestingLevel.Next())
        let fx = f x
        let r = fun v -> fx |> DiffSharp.reverse v; x.derivative
        fx.primal, r
    static member evalForwardDiffs (f:Tensor->Tensor) x (v:Tensor[]) =
        let n = v.Length
        if n = 0 then [|f x|]
        else
            let mutable x = x
            for i in 0..n-1 do
                x <- x |> DiffSharp.forwardDiff (GlobalNestingLevel.Next()) v.[i]
            let mutable fx = f x
            [|for _ in 0..n-1 do
                let d = fx.derivativeDeep
                fx <- fx.primal
                d
                |] |> Array.rev |> Array.append [|fx|]
    static member pjacobianv f (x:Tensor) v = 
        let fx, d = DiffSharp.evalForwardDiff f x v
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, d
    static member jacobianv f x v = DiffSharp.pjacobianv f x v |> snd
    static member pgradv f (x:Tensor) v =
        let fx, d = DiffSharp.evalForwardDiff f x v
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, d
    static member gradv f x v = DiffSharp.pgradv f x v |> snd
    static member pdiff f (x:Tensor) =
        let fx, d = DiffSharp.evalForwardDiff f x (x.onesLike())
        if x.dim > 0 then failwithf "f must be a function of a scalar, encountered f:%A->%A" x.shape fx.shape
        fx, d
    static member diff f x = DiffSharp.pdiff f x |> snd
    static member ppdiffn (n:int) (f:Tensor->Tensor) (x:Tensor) =
        if n < 0 then failwith "Differentiation order n must be >= 0"
        if x.dim > 0 then failwithf "f must be a function of a scalar"
        DiffSharp.evalForwardDiffs f x (Array.create n (x.onesLike()))
    static member pdiffn n f x = let a = DiffSharp.ppdiffn n f x in a |> Array.head, a |> Array.last
    static member diffn n f x = DiffSharp.pdiffn n f x |> snd
    static member pdiff2 f x = DiffSharp.pdiffn 2 f x
    static member diff2 f x = DiffSharp.diffn 2 f x
    static member pjacobianTv f x v =
        let fx, r = DiffSharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, r v
    static member jacobianTv f x v = DiffSharp.pjacobianTv f x v |> snd
    static member pjacobian (f:Tensor->Tensor) x =
        let fx, r = DiffSharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        if x.nelement > fx.nelement then
            fx, DiffSharp.stack(Array.init fx.nelement (fun i -> r (x.onehotLike(fx.nelement, i))), 0)
        else
            fx, DiffSharp.stack(Array.init x.nelement (fun j -> DiffSharp.jacobianv f x (x.onehotLike(x.nelement, j))), 1)
    static member jacobian f x = DiffSharp.pjacobian f x |> snd
    static member pgrad f x =
        let fx, r = DiffSharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, r (fx.onesLike())
    static member grad f x = DiffSharp.pgrad f x |> snd
    static member pgradhessianv f x v =
        let x = x |> DiffSharp.reverseDiff (GlobalNestingLevel.Next())
        let fx, gv = DiffSharp.pgradv f x v
        gv.reverse()
        fx.primal, gv.primal, x.derivative
    static member gradhessianv f x v = let _, gv, hv = DiffSharp.pgradhessianv f x v in gv, hv
    static member phessianv f x v = let fx, _, hv = DiffSharp.pgradhessianv f x v in fx, hv
    static member hessianv f x v = DiffSharp.phessianv f x v |> snd
    static member phessian (f:Tensor->Tensor) (x:Tensor) =
        let mutable fx = DiffSharp.zero()
        let h = DiffSharp.stack(Array.init x.nelement (fun j -> let ffxx, hv = DiffSharp.phessianv f x (x.onehotLike(x.nelement, j)) in fx <- ffxx; hv), 1)
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, h
    static member hessian f x = DiffSharp.phessian f x |> snd


type dsharp = DiffSharp