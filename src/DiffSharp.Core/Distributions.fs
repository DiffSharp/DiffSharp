namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util
open System.Collections.Generic


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
type Distribution<'T>() =
    abstract member sample: unit -> 'T
    abstract member logprob: 'T -> Tensor


[<AbstractClass>]
type TensorDistribution() =
    inherit Distribution<Tensor>()
    member d.sample(numSamples:int) = Array.init numSamples (fun _ -> d.sample()) |> dsharp.stack
    abstract member batchShape: int[]
    abstract member eventShape: int[]
    abstract member mean: Tensor
    abstract member stddev: Tensor
    abstract member variance: Tensor
    default d.stddev = d.variance.sqrt()
    default d.variance = d.stddev * d.stddev
    member d.prob(value) = d.logprob(value).exp()


type Normal(mean:Tensor, stddev:Tensor) =
    inherit TensorDistribution()
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
    inherit TensorDistribution()
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
    inherit TensorDistribution()
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
    inherit TensorDistribution()
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


type Empirical<'T when 'T:equality>(values:seq<'T>, ?weights:Tensor, ?logWeights:Tensor, ?combineDuplicates:bool) =
    inherit Distribution<'T>()
    let _categorical, _weighted =
        match weights, logWeights with
        | Some _, Some _ -> failwithf "Expecting only one of weights, logWeights"
        | Some w, None -> Categorical(probs=w), true
        | None, Some lw -> Categorical(logits=lw), true
        | None, None -> Categorical(probs=dsharp.ones([values |> Seq.length])), false  // Uniform weights for unweighted distributions
    let mutable _categorical = _categorical
    let mutable _weighted = _weighted
    let mutable _values = values |> Array.ofSeq
    let _valuesTensor =
            lazy(try _values |> Array.map (fun v -> box v :?> Tensor) |> dsharp.stack
                    with | _ -> 
                        try _values |> Array.map (dsharp.tensor) |> dsharp.stack
                        with | _ -> failwith "Not supported because Empirical does not hold values that are Tensors or can be converted to Tensors")
    do
        let combineDuplicates = defaultArg combineDuplicates false
        if combineDuplicates then
            let newValues, newLogWeights = 
                if _weighted then
                    let uniques = Dictionary<'T, Tensor>()
                    for i = 0 to _values.Length-1 do
                        let v, lw = _values.[i], _categorical.logits.[i]
                        if uniques.ContainsKey(v) then
                            let lw2 = uniques.[v]
                            uniques.[v] <- dsharp.stack([lw; lw2]).logsumexp(dim=0)
                        else uniques.[v] <- lw
                    copyKeys uniques, dsharp.stack(copyValues uniques).view(-1)
                else
                    let vals, counts = getUniqueCounts _values false
                    vals, dsharp.tensor(counts)
            _values <- newValues
            _categorical <- Categorical(logits=newLogWeights)
        _weighted <- not (allEqual _categorical.probs)
    member d.values = _values
    member d.valuesTensor = _valuesTensor.Force()
    member d.length = d.values.Length
    member d.weights = _categorical.probs
    member d.logWeights = _categorical.logits
    member d.isWeighted = _weighted
    member d.Item
        with get(i) = d.values.[i], d.weights.[i]
    member d.GetSlice(start, finish) =
        let start = defaultArg start 0
        let finish = defaultArg finish d.length - 1
        Empirical(d.values.[start..finish], logWeights=d.logWeights.[start..finish])
    member d.unweighted() = Empirical(d.values)
    member d.map (f:'T->'a) = Empirical(Array.map f d.values, logWeights=d.logWeights)
    member d.filter (predicate:'T->bool) =
        let results = ResizeArray<'T*Tensor>()
        Array.iteri (fun i v -> if predicate v then results.Add(v, d.logWeights.[i])) d.values
        let v, lw = results.ToArray() |> Array.unzip
        Empirical(v, logWeights=dsharp.stack(lw))
    member d.sample(?minIndex:int, ?maxIndex:int) = // minIndex is inclusive, maxIndex is exclusive
        let minIndex = defaultArg minIndex 0
        let maxIndex = defaultArg maxIndex d.length
        if minIndex <> 0 || maxIndex <> d.length then
            if minIndex < 0 || minIndex > d.length then failwithf "Expecting 0 <= minIndex (%A) <= %A" minIndex d.length
            if maxIndex < 0 || maxIndex > d.length then failwithf "Expecting 0 <= maxIndex (%A) <= %A" maxIndex d.length
            if minIndex >= maxIndex then failwithf "Expecting minIndex (%A) < maxIndex (%A)" minIndex maxIndex
            d.[minIndex..maxIndex].sample()
        else
            let i = _categorical.sample() |> int in d.values.[i]
    member d.resample(numSamples, ?minIndex:int, ?maxIndex:int) = Array.init numSamples (fun _ -> d.sample(?minIndex=minIndex, ?maxIndex=maxIndex)) |> Empirical
    member d.thin(numSamples, ?minIndex:int, ?maxIndex:int) = 
        if d.isWeighted then failwithf "Cannot thin weighted distribution. Consider transforming Empirical to an unweigted Empirical by resampling."
        let minIndex = defaultArg minIndex 0
        let maxIndex = defaultArg maxIndex d.length
        let step = max 1 (int(floor (float (maxIndex - minIndex)) / (float numSamples)))
        let results = ResizeArray<'T>()
        for i in 0..step..d.length-1 do
            let v = d.values.[i]
            results.Add(v)
        Empirical(results.ToArray())
    member d.combineDuplicates() = Empirical(d.values, logWeights=d.logWeights, combineDuplicates=true)
    member d.expectation (f:Tensor->Tensor) =
        if d.isWeighted then d.valuesTensor |> Seq.mapi (fun i v -> d.weights.[i]*(f v)) |> dsharp.stack |> dsharp.sum(0)
        else d.valuesTensor |> Seq.map f |> dsharp.stack |> dsharp.mean(0)
    member d.mean = d.expectation(id)
    member d.variance = let mean = d.mean in d.expectation(fun x -> (x-mean)**2)
    member d.stddev = dsharp.sqrt(d.variance)
    member d.mode =
        if d.isWeighted then 
            let dCombined = d.combineDuplicates()
            let i = dCombined.logWeights.argmax() in dCombined.values.[i.[0]]
        else
            let vals, _ = getUniqueCounts d.values true
            vals.[0]
    member d.min = d.valuesTensor.min()
    member d.max = d.valuesTensor.max()
    member d.effectiveSampleSize = 1. / d.weights.pow(2).sum()
    override d.sample() = d.sample(minIndex=0, maxIndex=d.length)
    override d.logprob(_) = failwith "Not supported"  // TODO: can be implemented using density estimation
    override d.ToString() = sprintf "Empirical(length:%A)" d.length