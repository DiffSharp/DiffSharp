namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type Distribution() =
    abstract member Sample: unit -> Tensor
    member d.Sample(?numSamples:int) =
        let numSamples = defaultArg numSamples -1
        if numSamples = -1 then
            d.Sample()
        else
            List.init numSamples (fun _ -> d.Sample()) |> Tensor.Stack
    abstract member GetString: unit -> string
    override t.ToString() = t.GetString()
    abstract member Mean: Tensor
    abstract member Stddev : Tensor
    member d.Variance = d.Stddev * d.Stddev
    abstract member Logprob: Tensor -> Tensor
    member d.Prob(value) = Tensor.Exp(d.Logprob(value))

type Uniform(low:Tensor, high:Tensor) =
    inherit Distribution()
    do if low.Shape <> high.Shape then failwithf "Expecting low and high with the same shape, received %A, %A" low.Shape high.Shape
    do if low.Dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    member d.Low = low
    member d.High = high
    member private d.Range = high - low
    override d.Mean = (low + high) / 2.
    override d.Stddev = d.Range * d.Range / 12.f
    override d.Sample() = d.Low + Tensor.RandomLike(d.Low) * d.Range
    override d.Logprob(value) = 
        if value.Shape <> d.Mean.Shape then invalidArg "value" <| sprintf "Expecting a value with shape %A, received %A" d.Low.Shape value.Shape
        let lb = d.Low.Le(value)
        let ub = d.High.Gt(value)
        log (lb * ub) - log d.Range
    override d.GetString() = sprintf "Uniform(low:%A, high:%A)" d.Low d.High

type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.Shape <> stddev.Shape then failwithf "Expecting mean and standard deviation with the same shape, received %A, %A" mean.Shape stddev.Shape
    do if mean.Dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    override d.Mean = mean
    override d.Stddev = stddev
    override d.Sample() = d.Mean + Tensor.RandomNormalLike(d.Mean) * d.Stddev
    override d.Logprob(value) = 
        if value.Shape <> d.Mean.Shape then invalidArg "value" <| sprintf "Expecting a value with shape %A, received %A" d.Mean.Shape value.Shape
        let v = value - d.Mean in -(v * v) / (2. * d.Variance) - (log d.Stddev) - logSqrt2Pi
    override d.GetString() = sprintf "Normal(mean:%A, stddev:%A)" d.Mean d.Stddev
    