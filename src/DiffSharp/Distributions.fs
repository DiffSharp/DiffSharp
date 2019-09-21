namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type Distribution() =
    abstract member Sample: int -> Tensor
    abstract member Mean: Tensor
    abstract member Stddev : Tensor

type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.Shape <> stddev.Shape then failwithf "Expecting mean and standard deviation with the same shape, received %A, %A" mean.Shape stddev.Shape
    override d.Mean = mean
    override d.Stddev = stddev

    override d.Sample(numSamples:int) =
        // let numSamples = defaultArg numSamples 1
        d.Mean + Tensor.RandomNormalLike(d.Mean) * d.Stddev
