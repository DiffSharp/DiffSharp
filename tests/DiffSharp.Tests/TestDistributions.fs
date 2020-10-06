namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =
    let numEmpiricalSamples = 1000

    [<SetUp>]
    member _.Setup () =
        dsharp.seed(123)
        ()

    [<Test>]
    member _.TestDistributionsNormal () =
        for combo in Combos.AllDevicesAndBackends do
            let meanCorrect = combo.tensor(10.)
            let stddevCorrect = combo.tensor(3.5)
            let d = Normal(meanCorrect, stddevCorrect)
            let batchShape = d.batchShape
            let batchShapeCorrect = [||]
            let eventShape = d.eventShape
            let eventShapeCorrect = [||]
            let mean = d.mean
            let stddev = d.stddev
            let samples = d.sample(numEmpiricalSamples)
            let samplesMean = samples.mean()
            let samplesStddev = samples.stddev()
            let logprob = d.logprob(combo.tensor(20.))
            let logprobCorrect = combo.tensor(-6.2533)

            Assert.CheckEqual(batchShapeCorrect, batchShape)
            Assert.CheckEqual(eventShapeCorrect, eventShape)
            Assert.CheckEqual(meanCorrect, mean)
            Assert.CheckEqual(stddevCorrect, stddev)
            Assert.True(samplesMean.allclose(meanCorrect, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1))

    [<Test>]
    member _.TestDistributionsNormalBatched () =
        for combo in Combos.AllDevicesAndBackends do
            let meanCorrect = combo.tensor([10., 20.])
            let stddevCorrect = combo.tensor([3.5, 1.2])
            let d = Normal(meanCorrect, stddevCorrect)
            let batchShape = d.batchShape
            let batchShapeCorrect = [|2|]
            let eventShape = d.eventShape
            let eventShapeCorrect = [||]
            let mean = d.mean
            let stddev = d.stddev
            let samples = d.sample(numEmpiricalSamples)
            let samplesMean = samples.mean(0)
            let samplesStddev = samples.stddev(0)
            let logprob = d.logprob(combo.tensor([20., 21.]))
            let logprobCorrect = combo.tensor([-6.2533, -1.4485])

            Assert.CheckEqual(batchShapeCorrect, batchShape)
            Assert.CheckEqual(eventShapeCorrect, eventShape)
            Assert.CheckEqual(meanCorrect, mean)
            Assert.CheckEqual(stddevCorrect, stddev)            
            Assert.True(samplesMean.allclose(meanCorrect, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1))

    [<Test>]
    member _.TestDistributionsUniform () =
        for combo in Combos.AllDevicesAndBackends do
            let lowCorrect = combo.tensor(0.5)
            let highCorrect = combo.tensor(10.5)
            let rangeCorrect = highCorrect - lowCorrect
            let d = Uniform(lowCorrect, highCorrect)
            let batchShape = d.batchShape
            let batchShapeCorrect = [||]
            let eventShape = d.eventShape
            let eventShapeCorrect = [||]
            let low = d.low
            let high = d.high
            let range = d.range
            let mean = d.mean
            let stddev = d.stddev
            let samples = d.sample(numEmpiricalSamples)
            let samplesMean = samples.mean()
            let samplesStddev = samples.stddev()
            let meanCorrect = combo.tensor(5.5)
            let stddevCorrect = combo.tensor(2.8868)
            let logprob = d.logprob(combo.tensor(8.))
            let logprobCorrect = combo.tensor(-2.3026)

            Assert.CheckEqual(batchShapeCorrect, batchShape)
            Assert.CheckEqual(eventShapeCorrect, eventShape)
            Assert.CheckEqual(lowCorrect, low)
            Assert.CheckEqual(highCorrect, high)
            Assert.CheckEqual(rangeCorrect, range)
            Assert.True(mean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(stddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsUniformBatched () =
        for combo in Combos.AllDevicesAndBackends do
            let lowCorrect = combo.tensor([0.5, 0., -5.])
            let highCorrect = combo.tensor([10.5, 1., 5.])
            let rangeCorrect = highCorrect - lowCorrect
            let d = Uniform(lowCorrect, highCorrect)
            let batchShape = d.batchShape
            let batchShapeCorrect = [|3|]
            let eventShape = d.eventShape
            let eventShapeCorrect = [||]
            let low = d.low
            let high = d.high
            let range = d.range
            let mean = d.mean
            let stddev = d.stddev
            let samples = d.sample(numEmpiricalSamples)
            let samplesMean = samples.mean(0)
            let samplesStddev = samples.stddev(0)
            let meanCorrect = combo.tensor([5.5, 0.5, 0.])
            let stddevCorrect = combo.tensor([2.8868, 0.28867, 2.88435])
            let logprob = d.logprob(combo.tensor([8., 0.2, 4.]))
            let logprobCorrect = combo.tensor([-2.3026, 0., -2.3026])

            Assert.CheckEqual(batchShapeCorrect, batchShape)
            Assert.CheckEqual(eventShapeCorrect, eventShape)
            Assert.CheckEqual(lowCorrect, low)
            Assert.CheckEqual(highCorrect, high)
            Assert.CheckEqual(rangeCorrect, range)
            Assert.True(mean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(stddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsBernoulli () =
        for combo in Combos.AllDevicesAndBackends do
            for logit in [false; true] do
                let probsCorrect, logitsCorrect, d =
                    if logit then
                        let logits = combo.tensor(-1.6094)
                        let d = Bernoulli(logits=logits)
                        let probs = logitsToProbs logits true
                        probs, logits, d
                    else
                        let probs = combo.tensor(0.2)
                        let d = Bernoulli(probs=probs)
                        let logits = probsToLogits probs true
                        probs, logits, d
                let batchShape = d.batchShape
                let batchShapeCorrect = [||]
                let eventShape = d.eventShape
                let eventShapeCorrect = [||]
                let probs = d.probs
                let logits = d.logits
                let samples = d.sample(numEmpiricalSamples)
                let samplesMean = samples.mean(0)
                let samplesStddev = samples.stddev(0)
                let meanCorrect = combo.tensor(0.2)
                let stddevCorrect = combo.tensor(0.4)
                let logprob = d.logprob(combo.tensor(0.2))
                let logprobCorrect = combo.tensor(-0.5004)

                Assert.CheckEqual(batchShapeCorrect, batchShape)
                Assert.CheckEqual(eventShapeCorrect, eventShape)
                Assert.True(probsCorrect.allclose(probs, 0.1, 0.1))
                Assert.True(logitsCorrect.allclose(logits, 0.1, 0.1))
                Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
                Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
                Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsBernoulliBatched () =
        for combo in Combos.AllDevicesAndBackends do
            for logit in [false; true] do
                let probsCorrect, logitsCorrect, d =
                    if logit then
                        let logits = combo.tensor([[ 0.0000, -1.3863, -0.8473],
                                                    [-1.3863, -1.3863,  0.4055]])
                        let d = Bernoulli(logits=logits)
                        let probs = logitsToProbs logits true
                        probs, logits, d
                    else
                        let probs = combo.tensor([[0.5, 0.2, 0.3], 
                                                    [0.2, 0.2, 0.6]])
                        let d = Bernoulli(probs=probs)
                        let logits = probsToLogits probs true
                        probs, logits, d
                let batchShape = d.batchShape
                let batchShapeCorrect = probsCorrect.shape
                let eventShape = d.eventShape
                let eventShapeCorrect = [||]
                let probs = d.probs
                let logits = d.logits
                let samples = d.sample(10*numEmpiricalSamples)
                let samplesMean = samples.mean(0)
                let samplesStddev = samples.stddev(0)
                let meanCorrect = probs
                let stddevCorrect = combo.tensor([[0.5000, 0.4000, 0.4583],
                                                    [0.4000, 0.4000, 0.4899]])
                let logprob = d.logprob(combo.tensor([[1., 0., 1.],
                                                        [0., 1., 0.]]))
                let logprobCorrect = combo.tensor([[-0.6931, -0.2231, -1.2040],
                                                    [-0.2231, -1.6094, -0.9163]])

                Assert.CheckEqual(batchShapeCorrect, batchShape)
                Assert.CheckEqual(eventShapeCorrect, eventShape)
                Assert.True(probsCorrect.allclose(probs, 0.1, 0.1))
                Assert.True(logitsCorrect.allclose(logits, 0.1, 0.1))
                Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
                Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
                Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsCategorical () =
        for combo in Combos.AllDevicesAndBackends do
            for logit in [false; true] do
                let probsCorrect, logitsCorrect, d =
                    if logit then
                        let logits = combo.tensor([-2.30259, -1.60944, -0.356675])
                        let d = Categorical(logits=logits)
                        let probs = logitsToProbs logits false
                        probs, logits, d
                    else
                        let probs = combo.tensor([1, 2, 7])  // Gets normalized to [0.1, 0.2, 0.7]
                        let d =  Categorical(probs=probs)
                        let probs = probs / probs.sum()
                        let logits = probsToLogits probs false
                        probs, logits, d
                let batchShape = d.batchShape
                let batchShapeCorrect = [||]
                let eventShape = d.eventShape
                let eventShapeCorrect = [||]
                let probs = d.probs
                let logits = d.logits
                let samples = d.sample(numEmpiricalSamples).cast(combo.dtype)
                let samplesMean = samples.mean(0)
                let samplesStddev = samples.stddev(0)
                let meanCorrect = combo.tensor(1.6)
                let stddevCorrect = combo.tensor(0.666)
                let logprob = d.logprob(combo.tensor(0))
                let logprobCorrect = combo.tensor(-2.30259)

                Assert.CheckEqual(batchShapeCorrect, batchShape)
                Assert.CheckEqual(eventShapeCorrect, eventShape)
                Assert.True(probsCorrect.allclose(probs, 0.1, 0.1))
                Assert.True(logitsCorrect.allclose(logits, 0.1, 0.1))
                Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
                Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
                Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsCategoricalBatched () =
        for combo in Combos.AllDevicesAndBackends do
            let probsCorrect = combo.tensor([[0.1, 0.2, 0.7],
                                                [0.2, 0.5, 0.3]])
            let logitsCorrect = probsToLogits probsCorrect false
            let d = Categorical(probs=probsCorrect)
            let batchShape = d.batchShape
            let batchShapeCorrect = [|2|]
            let eventShape = d.eventShape
            let eventShapeCorrect = [||]
            let probs = d.probs
            let logits = d.logits
            let samples = d.sample(numEmpiricalSamples).cast(combo.dtype)
            let samplesMean = samples.mean(0)
            let samplesStddev = samples.stddev(0)
            let meanCorrect = combo.tensor([1.6, 1.1])
            let stddevCorrect = combo.tensor([0.666, 0.7])
            let logprob = d.logprob(combo.tensor([0, 1]))
            let logprobCorrect = combo.tensor([-2.30259, -0.693147])

            Assert.CheckEqual(batchShapeCorrect, batchShape)
            Assert.CheckEqual(eventShapeCorrect, eventShape)
            Assert.True(probsCorrect.allclose(probs, 0.1, 0.1))
            Assert.True(logitsCorrect.allclose(logits, 0.1, 0.1))
            Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

    [<Test>]
    member _.TestDistributionsEmpirical () =
        for combo in Combos.AllDevicesAndBackends do
            let values = combo.tensor([1,2,3])
            let logWeights = combo.tensor([1,2,3])

            let dist = Empirical(values.unstack(), logWeights=logWeights)
            let distMean = dist.mean
            let distStddev = dist.stddev
            let distMeanCorrect = combo.tensor(2.575210)
            let distStddevCorrect = combo.tensor(0.651463)
            let distEffectiveSampleSize = dist.effectiveSampleSize
            let distEffectiveSampleSizeCorrect = combo.tensor(1.9587)
            let distMin = dist.min
            let distMax = dist.max
            let distMinCorrect = combo.tensor(1)
            let distMaxCorrect = combo.tensor(3)
            let distExpectationSin = dist.expectation(dsharp.sin)
            let distExpectationSinCorrect = combo.tensor(0.392167)
            let distMapSin = dist.map(dsharp.sin)
            let distMapSinMean = distMapSin.mean
            let distMapSinMeanCorrect = combo.tensor(0.392167)

            let distEmpirical = Empirical([for _ = 0 to numEmpiricalSamples do dist.sample()])
            let distEmpiricalMean = distEmpirical.mean
            let distEmpiricalStddev = distEmpirical.stddev

            let distUnweighted = dist.unweighted()
            let distUnweightedMean = distUnweighted.mean
            let distUnweightedStddev = distUnweighted.stddev
            let distUnweightedMeanCorrect = combo.tensor(2.)
            let distUnweightedStddevCorrect = combo.tensor(0.816497)

            Assert.True(distMeanCorrect.allclose(distMean, 0.1))
            Assert.True(distStddevCorrect.allclose(distStddev, 0.1))
            Assert.True(distEffectiveSampleSizeCorrect.allclose(distEffectiveSampleSize, 0.1))
            Assert.CheckEqual(distMinCorrect, distMin)
            Assert.CheckEqual(distMaxCorrect, distMax)
            Assert.True(distExpectationSinCorrect.allclose(distExpectationSin, 0.1))
            Assert.True(distMapSinMeanCorrect.allclose(distMapSinMean, 0.1))
            Assert.True(distMeanCorrect.allclose(distEmpiricalMean, 0.1))
            Assert.True(distStddevCorrect.allclose(distEmpiricalStddev, 0.1))
            Assert.True(distUnweightedMeanCorrect.allclose(distUnweightedMean, 0.1))
            Assert.True(distUnweightedStddevCorrect.allclose(distUnweightedStddev, 0.1))

    [<Test>]
    member _.TestDistributionsEmpiricalCombineDuplicatesMode () =
        for combo in Combos.AllDevicesAndBackends do
            let values = combo.tensor([0,1,2,2,2,2,3,4,4,5,5,5,6,7,7,8,9])
            let weights = combo.tensor([0.0969, 0.1948, 0.7054, 0.0145, 0.7672, 0.1592, 0.4845, 0.7710, 0.3588, 0.8622, 0.7621, 0.6102, 0.9421, 0.0774, 0.8294, 0.7371, 0.3742])

            let dist = Empirical(values.unstack(), weights=weights)
            let distMode = dist.mode
            let distModeCorrect = combo.tensor(5)
            let distLength = dist.length
            let distLengthCorrect = 17
            Assert.CheckEqual(distModeCorrect, distMode)
            Assert.CheckEqual(distLengthCorrect, distLength)

            let distCombined = Empirical(values.unstack(), weights=weights).combineDuplicates()
            let distCombinedMode = distCombined.mode
            let distCombinedModeCorrect = combo.tensor(5)
            let distCombinedLength = distCombined.length
            let distCombinedLengthCorrect = 10
            Assert.CheckEqual(distCombinedModeCorrect, distCombinedMode)
            Assert.CheckEqual(distCombinedLengthCorrect, distCombinedLength)

            let distUnweighted = Empirical(values.unstack())
            let distUnweightedMode = distUnweighted.mode
            let distUnweightedModeCorrect = combo.tensor(2)
            let distUnweightedLength = distUnweighted.length
            let distUnweightedLengthCorrect = 17
            Assert.CheckEqual(distUnweightedModeCorrect, distUnweightedMode)
            Assert.CheckEqual(distUnweightedLengthCorrect, distUnweightedLength)

            let distUnweightedCombined = Empirical(values.unstack()).combineDuplicates()
            let distUnweightedCombinedMode = distUnweightedCombined.mode
            let distUnweightedCombinedModeCorrect = combo.tensor(2)
            let distUnweightedCombinedLength = distUnweightedCombined.length
            let distUnweightedCombinedLengthCorrect = 10
            Assert.CheckEqual(distUnweightedCombinedModeCorrect, distUnweightedCombinedMode)
            Assert.CheckEqual(distUnweightedCombinedLengthCorrect, distUnweightedCombinedLength)

    [<Test>]
    member _.TestDistributionsEmpiricalResampleFilter () =
        for combo in Combos.AllDevicesAndBackends do
            let values = combo.tensor([0,1,2,3,4,5])
            let weights = combo.tensor([0.0969, 0.1948, 0.7054, 0.0145, 0.7672, 0.1592])

            let dist = Empirical(values.unstack(), weights=weights)
            let distMean = dist.mean
            let distStddev = dist.stddev
            let distMeanCorrect = combo.tensor(2.8451)
            let distStddevCorrect = combo.tensor(1.3844)
            let distWeighted = dist.isWeighted
            let distWeightedCorrect = true

            let distResampled = dist.resample(numEmpiricalSamples)
            let distResampledMean = distResampled.mean
            let distResampledStddev = distResampled.stddev
            let distResampledWeighted = distResampled.isWeighted
            let distResampledWeightedCorrect = false

            Assert.True(distMeanCorrect.allclose(distMean, 0.1))
            Assert.True(distStddevCorrect.allclose(distStddev, 0.1))
            Assert.CheckEqual(distWeightedCorrect, distWeighted)
            Assert.True(distMeanCorrect.allclose(distResampledMean, 0.1))
            Assert.True(distStddevCorrect.allclose(distResampledStddev, 0.1))
            Assert.CheckEqual(distResampledWeightedCorrect, distResampledWeighted)

            let distResampledMinMax = dist.resample(numEmpiricalSamples, minIndex=1, maxIndex=3)
            let distResampledMinMaxMean = distResampledMinMax.mean
            let distResampledMinMaxStddev = distResampledMinMax.stddev
            let distResampledMinMaxMeanCorrect = combo.tensor(1.7802)
            let distResampledMinMaxStddevCorrect = combo.tensor(0.4141)            
            let distResampledMinMaxWeighted = distResampledMinMax.isWeighted
            let distResampledMinMaxWeightedCorrect = false

            Assert.True(distResampledMinMaxMeanCorrect.allclose(distResampledMinMaxMean, 0.1))
            Assert.True(distResampledMinMaxStddevCorrect.allclose(distResampledMinMaxStddev, 0.1))
            Assert.CheckEqual(distResampledMinMaxWeightedCorrect, distResampledMinMaxWeighted)

            let distFiltered = dist.filter(fun v -> v > combo.tensor(0) && v < combo.tensor(3))
            let distFilteredMean = distFiltered.mean
            let distFilteredStddev = distFiltered.stddev
            let distFilteredMeanCorrect = combo.tensor(1.7802)
            let distFilteredStddevCorrect = combo.tensor(0.4141)            
            let distFilteredWeighted = distFiltered.isWeighted
            let distFilteredWeightedCorrect = true

            Assert.True(distFilteredMeanCorrect.allclose(distFilteredMean, 0.1))
            Assert.True(distFilteredStddevCorrect.allclose(distFilteredStddev, 0.1))
            Assert.CheckEqual(distFilteredWeightedCorrect, distFilteredWeighted)

    [<Test>]
    member _.TestDistributionsEmpiricalThin () =
        let values = [1;2;3;4;5;6;7;8;9;10;11;12]
        let dist = Empirical(values)
        let distThinned = dist.thin(4)
        let distThinnedValues = distThinned.values
        let distThinnedValuesCorrect = [| 1;4;7;10 |]
        Assert.CheckEqual(distThinnedValuesCorrect, distThinnedValues)