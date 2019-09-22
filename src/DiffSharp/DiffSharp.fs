namespace DiffSharp
open DiffSharp.Util

module DiffSharp =
    let inline Seed(seed) = Random.Seed(seed)
    let inline Nest(level) = GlobalNestingLevel.Set(level)
    let inline NestReset() = GlobalNestingLevel.Reset()
    let inline NestNext() = GlobalNestingLevel.Next() |> ignore
    let inline primal (tensor:Tensor) = tensor.Primal
    let inline derivative (tensor:Tensor) = tensor.Derivative
    let inline primalDerivative tensor = tensor |> primal, tensor |> derivative
    let inline makeForward (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.GetForward(derivative, tag)
    let inline makeReverse (tag:uint32) (tensor:Tensor) = tensor.GetReverse(tag)
    let inline reverseReset (tensor:Tensor) = tensor.ReverseReset()
    let inline reversePush (value:Tensor) (tensor:Tensor) = tensor.ReversePush(value)
    let inline reverseProp (value:Tensor) (tensor:Tensor) = tensor |> reverseReset; tensor |> reversePush value
    let inline jacobianv' f x v = x |> makeForward (GlobalNestingLevel.Next()) v |> f |> primalDerivative
    let inline jacobianv f x v = jacobianv' f x v |> snd
    let inline jacobianTv'' f x =
        let xa = x |> makeReverse (GlobalNestingLevel.Next())
        let z = f xa
        let zp = z |> primal
        let r =
            fun v ->
                z |> reverseProp v
                xa |> derivative
        zp, r
    let inline jacobianTv' f x v = let zp, r = jacobianTv'' f x in zp, r v
    let inline jacobianTv f x v = jacobianTv' f x v |> snd
    let inline gradv f x v = jacobianv f x v
    let inline gradv' f x v = jacobianv' f x v
    let inline diff' f x = jacobianv' f x (x |> Tensor.OnesLike)
    let inline diff f x = diff' f x |> snd
    let inline grad' f x = let zp, r = jacobianTv'' f x in zp, r (zp |> Tensor.OnesLike)
    let inline grad f x = grad' f x |> snd