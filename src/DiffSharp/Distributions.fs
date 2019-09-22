namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type Distribution() =
    abstract member Sample: ?numSamples:int -> Tensor
    abstract member Mean: Tensor
    abstract member Stddev : Tensor
    member d.Variance = d.Stddev * d.Stddev

type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.Shape <> stddev.Shape then failwithf "Expecting mean and standard deviation with the same shape, received %A, %A" mean.Shape stddev.Shape
    override d.Mean = mean
    override d.Stddev = stddev

    override d.Sample(?numSamples:int) =
        let numSamples = defaultArg numSamples -1
        if numSamples = -1 then
            d.Mean + Tensor.RandomNormalLike(d.Mean) * d.Stddev
        else
            List.init numSamples (fun _ -> d.Mean + Tensor.RandomNormalLike(d.Mean) * d.Stddev) |> Tensor.Stack
