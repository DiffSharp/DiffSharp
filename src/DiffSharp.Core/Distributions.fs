namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type Distribution() =
    abstract member Sample: unit -> Tensor
    abstract member Sample: int -> Tensor
    default d.Sample(numSamples:int) = Seq.init numSamples (fun _ -> d.Sample()) |> Tensor.stack
    abstract member GetString: unit -> string
    override t.ToString() = t.GetString()
    abstract member BatchShape: int[]
    abstract member EventShape: int[]
    abstract member Mean: Tensor
    abstract member Stddev: Tensor
    default d.Stddev = d.Variance |> Tensor.Sqrt
    abstract member Variance: Tensor
    default d.Variance = d.Stddev * d.Stddev
    abstract member Logprob: Tensor -> Tensor
    member d.Prob(value) = Tensor.Exp(d.Logprob(value))

type Uniform(low:Tensor, high:Tensor) =
    inherit Distribution()
    do if low.shape <> high.shape then failwithf "Expecting low and high with the same shape, received %A, %A" low.shape high.shape
    do if low.dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    member d.Low = low
    member d.High = high
    member private d.Range = high - low
    override d.BatchShape = low.shape
    override d.EventShape = [||]
    override d.Mean = (low + high) / 2.
    override d.Stddev = d.Range * d.Range / 12.
    override d.Sample() = d.Low + dsharp.randLike(d.Low) * d.Range
    override d.Logprob(value) = 
        if value.shape <> d.BatchShape then failwithf "Expecting a value with shape %A, received %A" d.BatchShape value.shape
        let lb = d.Low.le(value)
        let ub = d.High.gt(value)
        log (lb * ub) - log d.Range
    override d.GetString() = sprintf "Uniform(low:%A, high:%A)" d.Low d.High

type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.shape <> stddev.shape then failwithf "Expecting mean and standard deviation with the same shape, received %A, %A" mean.shape stddev.shape
    do if mean.dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    override d.BatchShape = d.Mean.shape
    override d.EventShape = [||]
    override d.Mean = mean
    override d.Stddev = stddev
    override d.Sample() = d.Mean + dsharp.randnLike(d.Mean) * d.Stddev
    override d.Logprob(value) = 
        if value.shape <> d.BatchShape then failwithf "Expecting a value with shape %A, received %A" d.BatchShape value.shape
        let v = value - d.Mean in -(v * v) / (2. * d.Variance) - (log d.Stddev) - logSqrt2Pi
    override d.GetString() = sprintf "Normal(mean:%A, stddev:%A)" d.Mean d.Stddev

type Categorical(?probs:Tensor, ?logprobs:Tensor) =
    inherit Distribution()
    member d.Probs =
        let probs =
            match probs with
            | None ->
                match logprobs with
                | None -> failwith "Expecting either probs or logprobs"
                | Some logprobs -> Tensor.Exp(logprobs)
            | Some probs -> probs
        if probs.dim < 1 || probs.dim > 2 then failwithf "Expecting a vector (1d) or batch of vector parameters (2d), received shape %A" probs.shape
        probs
    override d.BatchShape = if d.Probs.dim = 1 then [||] else [|d.Probs.shape.[0]|]
    override d.EventShape = [||]
    override d.Mean = dsharp.onesLike(d.Probs) * System.Double.NaN
    override d.Stddev = dsharp.onesLike(d.Probs) * System.Double.NaN
    override d.Sample(numSamples) =
        Tensor(d.Probs.primalRaw.RandomMultinomial(numSamples))
    override d.Sample() = d.Sample(1)
    override d.Logprob(value) =
        if value.shape <> d.BatchShape then failwithf "Expecting a value with shape %A, received %A" d.BatchShape value.shape
        if d.BatchShape.Length = 0 then
            let i = value.toScalar() |> toInt
            d.Probs.[i] |> dsharp.log
        else
            // let is:int[] = value.ToArray() :?> obj[] |> Array.map toInt
            Seq.init d.BatchShape.[0] (fun i -> d.Probs.[i]) |> Tensor.stack |> dsharp.log
    override d.GetString() = sprintf "Categorical(probs:%A)" d.Probs

// type Empirical(values:obj[], ?weights:Tensor, ?logweights:Tensor) =
//     inherit Distribution()
//     member d.Values = values
//     member d.Length = d.Values.Length
//     member d.Weights =
//         let weights =
//             match weights with
//             | None ->
//                 match logweights with
//                 | None -> failwith "Expecting either weights or logweights"
//                 | Some logweights -> Tensor.Exp(logweights)
//             | Some weights -> weights
//         if weights.dim <> 1 then failwithf "Expecting a vector (1d) of weights or logweights, received shape %A" weights.shape
//         weights
//     member d.Logweights = d.Weights |> dsharp.log
//     member d.Categorical = Categorical(probs=d.Weights)
//     override d.BatchShape = failwith "Not implemented"
//     override d.EventShape = failwith "Not implemented"
//     override d.Logprob(value) = failwith "Not implemented"
//     override d.Mean = d.Values |> Tensor.Create |> Tensor.Mean
//     override d.Variance = d.Values |> Tensor.Create |> Tensor.Variance
//     override d.Sample() = failwith "Not implemented"
//     override d.GetString() = sprintf "Empirical(length:%A)" d.Length