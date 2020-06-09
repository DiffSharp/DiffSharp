namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util


[<AbstractClass>]
type Distribution() =
    abstract member sample: unit -> Tensor
    default d.sample() = d.sample(1)
    abstract member sample: int -> Tensor
    default d.sample(numSamples:int) = Seq.init numSamples (fun _ -> d.sample()) |> dsharp.stack
    abstract member batchShape: int[]
    abstract member eventShape: int[]
    abstract member mean: Tensor
    abstract member stddev: Tensor
    default d.stddev = d.variance.sqrt()
    abstract member variance: Tensor
    default d.variance = d.stddev * d.stddev
    abstract member logprob: Tensor -> Tensor
    member d.prob(value) = d.logprob(value).exp()


type Uniform(low:Tensor, high:Tensor) =
    inherit Distribution()
    do if low.shape <> high.shape then failwithf "Expecting low and high with the same shape, received %A, %A" low.shape high.shape
    do if low.dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    member d.Low = low
    member d.High = high
    member private d.Range = high - low
    override d.batchShape = low.shape
    override d.eventShape = [||]
    override d.mean = (low + high) / 2.
    override d.stddev = d.Range * d.Range / 12.
    override d.sample() = d.Low + dsharp.randLike(d.Low) * d.Range
    override d.logprob(value) = 
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        let lb = low.le(value).cast(low.dtype)
        let ub = high.gt(value).cast(high.dtype)
        log (lb * ub) - log d.Range
    override d.ToString() = sprintf "Uniform(low:%A, high:%A)" d.Low d.High


type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.shape <> stddev.shape then failwithf "Expecting mean and standard deviation with the same shape, received %A, %A" mean.shape stddev.shape
    do if mean.dim > 1 then failwithf "Expecting scalar parameters (0D) or a batch of scalar parameters (1D)"
    override d.batchShape = d.mean.shape
    override d.eventShape = [||]
    override d.mean = mean
    override d.stddev = stddev
    override d.sample() = d.mean + dsharp.randnLike(d.mean) * d.stddev
    override d.logprob(value) = 
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        let v = value - d.mean in -(v * v) / (2. * d.variance) - (log d.stddev) - logSqrt2Pi
    override d.ToString() = sprintf "Normal(mean:%A, stddev:%A)" d.mean d.stddev


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
    override d.batchShape = if d.Probs.dim = 1 then [||] else [|d.Probs.shape.[0]|]
    override d.eventShape = [||]
    override d.mean = dsharp.onesLike(d.Probs) * System.Double.NaN
    override d.stddev = dsharp.onesLike(d.Probs) * System.Double.NaN
    override d.sample(numSamples) = dsharp.multinomial(d.Probs, numSamples)
    override d.logprob(value) =
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        if d.batchShape.Length = 0 then
            let i = value.toScalar() |> toInt
            d.Probs.[i] |> dsharp.log
        else
            // let is:int[] = value.ToArray() :?> obj[] |> Array.map toInt
            Seq.init d.batchShape.[0] (fun i -> d.Probs.[i]) |> Tensor.stack |> dsharp.log
    override d.ToString() = sprintf "Categorical(probs:%A)" d.Probs


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
//     override d.batchShape = failwith "Not implemented"
//     override d.eventShape = failwith "Not implemented"
//     override d.logprob(value) = failwith "Not implemented"
//     override d.mean = d.Values |> Tensor.Create |> Tensor.mean
//     override d.variance = d.Values |> Tensor.Create |> Tensor.variance
//     override d.sample() = failwith "Not implemented"
//     override d.ToString() = sprintf "Empirical(length:%A)" d.Length