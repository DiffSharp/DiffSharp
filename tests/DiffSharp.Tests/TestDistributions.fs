namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =
    let numEmpiricalSamples = 10000

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

