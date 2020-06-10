namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util


[<AutoOpen>]
module internal Utils = 
    let clampProbs probs = dsharp.clamp(probs, System.Double.Epsilon, 1. - System.Double.Epsilon)
    let probsToLogits probs isBinary = 
        let probsClamped = clampProbs probs
        if isBinary then dsharp.log(probsClamped) - dsharp.log(1. - probsClamped)
        else dsharp.log(probsClamped)
    let logitsToProbs logits isBinary = 
        if isBinary then dsharp.sigmoid(logits)
        elif logits.dim = 0 then logits.exp() else dsharp.softmax(logits, -1)


[<AbstractClass>]
type Distribution() =
    abstract member sample: unit -> Tensor
    default d.sample() = d.sample(1)
    abstract member sample: int -> Tensor
    default d.sample(numSamples:int) = Array.init numSamples (fun _ -> d.sample()) |> dsharp.stack
    abstract member batchShape: int[]
    abstract member eventShape: int[]
    abstract member mean: Tensor
    abstract member stddev: Tensor
    default d.stddev = d.variance.sqrt()
    abstract member variance: Tensor
    default d.variance = d.stddev * d.stddev
    abstract member logprob: Tensor -> Tensor
    member d.prob(value) = d.logprob(value).exp()


type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution()
    do if mean.shape <> stddev.shape then failwithf "Expecting mean and standard deviation with same shape, received %A, %A" mean.shape stddev.shape
    do if mean.dim > 1 then failwithf "Expecting scalar parameters (0-dimensional mean and stddev) or a batch of scalar parameters (1-dimensional mean and stddev)"
    override d.batchShape = d.mean.shape
    override d.eventShape = [||]
    override d.mean = mean
    override d.stddev = stddev
    override d.sample() = d.mean + dsharp.randnLike(d.mean) * d.stddev
    override d.logprob(value) = 
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        let v = value - d.mean in -(v * v) / (2. * d.variance) - (log d.stddev) - logSqrt2Pi
    override d.ToString() = sprintf "Normal(mean:%A, stddev:%A)" d.mean d.stddev


type Uniform(low:Tensor, high:Tensor) =
    inherit Distribution()
    do if low.shape <> high.shape then failwithf "Expecting low and high with same shape, received %A, %A" low.shape high.shape
    do if low.dim > 1 then failwithf "Expecting scalar parameters (0-dimensional low and high) or a batch of scalar parameters (1-dimensional low and high)"
    member d.low = low
    member d.high = high
    member d.range = high - low
    override d.batchShape = low.shape
    override d.eventShape = [||]
    override d.mean = (low + high) / 2.
    override d.variance = d.range * d.range / 12.
    override d.sample() = d.low + dsharp.randLike(d.low) * d.range
    override d.logprob(value) = 
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        let lb = low.le(value).cast(low.dtype)
        let ub = high.gt(value).cast(high.dtype)
        log (lb * ub) - log d.range
    override d.ToString() = sprintf "Uniform(low:%A, high:%A)" d.low d.high


type Bernoulli(?probs:Tensor, ?logits:Tensor) =
    inherit Distribution()
    let _probs, _logits, _dtype =
        match probs, logits with
        | Some _, Some _ -> failwithf "Expecting only one of probs, logits"
        | Some p, None -> let pp = p.float() in clampProbs pp, probsToLogits pp true, p.dtype  // Do not normalize probs
        | None, Some lp -> let lpp = lp.float() in logitsToProbs lpp true, lpp, lp.dtype  // Do not normalize logits
        | None, None -> failwithf "Expecting either probs or logits"
    member d.probs = _probs.cast(_dtype)
    member d.logits = _logits.cast(_dtype)
    override d.batchShape = d.probs.shape
    override d.eventShape = [||]
    override d.mean = d.probs
    override d.variance = (_probs * (1. - _probs)).cast(_dtype)
    override d.sample() = dsharp.bernoulli(_probs).cast(_dtype)
    override d.logprob(value) =
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        let lp = (_probs ** value) * ((1. - _probs) ** (1. - value))  // Correct but numerical stability can be improved
        lp.log().cast(_dtype)
    override d.ToString() = sprintf "Bernoulli(probs:%A)" d.probs


type Categorical(?probs:Tensor, ?logits:Tensor) =
    inherit Distribution()
    let _probs, _logits, _dtype =
        match probs, logits with
        | Some _, Some _ -> failwithf "Expecting only one of probs, logits"
        | Some p, None -> let pp = (p / p.sum(-1, keepDim=true)).float() in clampProbs pp, probsToLogits pp false, p.dtype  // Normalize probs
        | None, Some lp -> let lpp = (lp - lp.logsumexp(-1, keepDim=true)).float() in logitsToProbs lpp false, lpp, lp.dtype  // Normalize logits
        | None, None -> failwithf "Expecting either probs or logits"    
    do if _probs.dim < 1 || _probs.dim > 2 then failwithf "Expecting vector parameters (1-dimensional probs or logits) or batch of vector parameters (2-dimensional probs or logits), received shape %A" _probs.shape
    member d.probs = _probs.cast(_dtype)
    member d.logits = _logits.cast(_dtype)
    override d.batchShape = if d.probs.dim = 1 then [||] else [|d.probs.shape.[0]|]
    override d.eventShape = [||]
    override d.mean = dsharp.onesLike(d.probs) * System.Double.NaN
    override d.stddev = dsharp.onesLike(d.probs) * System.Double.NaN
    override d.sample() = dsharp.multinomial(_probs, 1).cast(_dtype).squeeze()
    override d.logprob(value) =
        if value.shape <> d.batchShape then failwithf "Expecting a value with shape %A, received %A" d.batchShape value.shape
        if d.batchShape.Length = 0 then
            let i = int value
            _logits.[i].cast(_dtype)
        else
            let is = value.int().toArray() :?> int[]
            let lp = Array.init d.batchShape.[0] (fun i -> _logits.[i, is.[i]]) |> dsharp.stack
            lp.cast(_dtype)
    override d.ToString() = sprintf "Categorical(probs:%A)" d.probs


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