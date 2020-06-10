namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =
    let numEmpiricalSamples = 5000

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

            Assert.AreEqual(batchShapeCorrect, batchShape)
            Assert.AreEqual(eventShapeCorrect, eventShape)
            Assert.AreEqual(meanCorrect, mean)
            Assert.AreEqual(stddevCorrect, stddev)
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

            Assert.AreEqual(batchShapeCorrect, batchShape)
            Assert.AreEqual(eventShapeCorrect, eventShape)
            Assert.AreEqual(meanCorrect, mean)
            Assert.AreEqual(stddevCorrect, stddev)            
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

            Assert.AreEqual(batchShapeCorrect, batchShape)
            Assert.AreEqual(eventShapeCorrect, eventShape)
            Assert.AreEqual(lowCorrect, low)
            Assert.AreEqual(highCorrect, high)
            Assert.AreEqual(rangeCorrect, range)
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

            Assert.AreEqual(batchShapeCorrect, batchShape)
            Assert.AreEqual(eventShapeCorrect, eventShape)
            Assert.AreEqual(lowCorrect, low)
            Assert.AreEqual(highCorrect, high)
            Assert.AreEqual(rangeCorrect, range)
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

                Assert.AreEqual(batchShapeCorrect, batchShape)
                Assert.AreEqual(eventShapeCorrect, eventShape)
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

                Assert.AreEqual(batchShapeCorrect, batchShape)
                Assert.AreEqual(eventShapeCorrect, eventShape)
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

                Assert.AreEqual(batchShapeCorrect, batchShape)
                Assert.AreEqual(eventShapeCorrect, eventShape)
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

            Assert.AreEqual(batchShapeCorrect, batchShape)
            Assert.AreEqual(eventShapeCorrect, eventShape)
            Assert.True(probsCorrect.allclose(probs, 0.1, 0.1))
            Assert.True(logitsCorrect.allclose(logits, 0.1, 0.1))
            Assert.True(samplesMean.allclose(meanCorrect, 0.1, 0.1))
            Assert.True(samplesStddev.allclose(stddevCorrect, 0.1, 0.1))
            Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))