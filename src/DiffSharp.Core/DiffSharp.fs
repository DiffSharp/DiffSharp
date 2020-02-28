namespace DiffSharp
open DiffSharp.Backend
open DiffSharp.Util

// Tensor operations
type DiffSharp =
    static member tensor(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) = Tensor.Create(value, ?dtype=dtype, ?device=device, ?backend=backend)
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
    static member extend(a:Tensor, shape:seq<int>) = a.extend(shape)
    static member stack(tensors:seq<Tensor>) = Tensor.stack(tensors)
    static member unstack(a:Tensor) = a.unstack()
    static member add(a:Tensor, b:Tensor) = a.add(b)
    static member sub(a:Tensor, b:Tensor) = a.sub(b)
    static member mul(a:Tensor, b:Tensor) = a.mul(b)
    static member div(a:Tensor, b:Tensor) = a.div(b)
    static member pow(a:Tensor, b:Tensor) = a.pow(b)
    static member matmul(a:Tensor, b:Tensor) = a.matmul(b)
    static member neg(a:Tensor) = a.neg()
    static member sum(a:Tensor) = a.sum()
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
    
// Functional differentiation API
type DiffSharp with
    static member seed(seed) = Random.Seed(seed)
    static member nest(level) = GlobalNestingLevel.Set(level)
    static member nestReset() = GlobalNestingLevel.Reset()
    static member nestNext() = GlobalNestingLevel.Next() |> ignore
    static member primal (tensor:Tensor) = tensor.Primal
    static member derivative (tensor:Tensor) = tensor.Derivative
    static member primalDerivative tensor = tensor |> DiffSharp.primal, tensor |> DiffSharp.derivative
    static member makeForward (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.ForwardDiff(derivative, tag)
    static member makeReverse (tag:uint32) (tensor:Tensor) = tensor.ReverseDiff(tag)
    static member reverseReset (tensor:Tensor) = tensor.ReverseReset(true)
    static member reversePush (value:Tensor) (tensor:Tensor) = tensor.ReversePush(value)
    static member reverseProp (value:Tensor) (tensor:Tensor) = tensor |> DiffSharp.reverseReset; tensor |> DiffSharp.reversePush value
    static member jacobianv' f x v = x |> DiffSharp.makeForward (GlobalNestingLevel.Next()) v |> f |> DiffSharp.primalDerivative
    static member jacobianv f x v = DiffSharp.jacobianv' f x v |> snd
    static member jacobianTv'' f x =
        let xa = x |> DiffSharp.makeReverse (GlobalNestingLevel.Next())
        let z = f xa
        let zp = z |> DiffSharp.primal
        let r =
            fun v ->
                z |> DiffSharp.reverseProp v
                xa |> DiffSharp.derivative
        zp, r
    static member jacobianTv' f x v = let zp, r = DiffSharp.jacobianTv'' f x in zp, r v
    static member jacobianTv f x v = DiffSharp.jacobianTv' f x v |> snd
    static member gradv f x v = DiffSharp.jacobianv f x v
    static member gradv' f x v = DiffSharp.jacobianv' f x v
    static member diff' f x = DiffSharp.jacobianv' f x (x |> Tensor.OnesLike)
    static member diff f x = DiffSharp.diff' f x |> snd
    static member grad' f x = let zp, r = DiffSharp.jacobianTv'' f x in zp, r (zp |> Tensor.OnesLike)
    static member grad f x = DiffSharp.grad' f x |> snd

type dsharp = DiffSharp