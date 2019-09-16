namespace DiffSharp
open DiffSharp.Util

module DiffSharp =
    let inline primal (tensor:Tensor) = tensor.Primal
    let inline derivative (tensor:Tensor) = tensor.Derivative
    let inline primalDerivative tensor = tensor |> primal, tensor |> derivative
    let inline makeForward (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.GetForward(derivative, tag)
    let inline makeReverse (tag:uint32) (tensor:Tensor) = tensor.GetReverse(tag)
    let inline reverseReset (tensor:Tensor) = tensor.ReverseReset()
    let inline reversePush (value:Tensor) (tensor:Tensor) = tensor.ReversePush(value)
    let inline reverseProp (value:Tensor) (tensor:Tensor) = tensor |> reverseReset; tensor |> reversePush value
    let inline jacobianv' f x v = x |> makeForward GlobalTagger.Next v |> f |> primalDerivative
    let inline jacobianv f x v = jacobianv' f x v |> snd
    let inline jacobianTv'' f x =
        let xa = x |> makeReverse GlobalTagger.Next
        let z = f xa
        let zp = z |> primal
        let r =
            fun v ->
                z |> reverseProp v
                xa |> derivative
        zp, r