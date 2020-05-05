namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =
    let numEmpiricalSamples = 10000

    [<SetUp>]
    member this.Setup () =
        dsharp.seed(123)
        ()

    [<Test>]
    member this.TestDistributionsNormal () =
        let meanCorrect = dsharp.tensor(10.)
        let stddevCorrect = dsharp.tensor(3.5)
        let d = Normal(meanCorrect, stddevCorrect)
        let batchShape = d.BatchShape
        let batchShapeCorrect = [||]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.mean()
        let stddev = samples.stddev()
        let logprob = d.Logprob(dsharp.tensor(20.))
        let logprobCorrect = dsharp.tensor(-6.2533)

        Assert.AreEqual(batchShapeCorrect, batchShape)
        Assert.AreEqual(eventShapeCorrect, eventShape)
        Assert.True(mean.allclose(meanCorrect, 0.1))
        Assert.True(stddev.allclose(stddevCorrect, 0.1))
        Assert.True(logprob.allclose(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsNormalBatched () =
        let meanCorrect = dsharp.tensor([10.; 20.])
        let stddevCorrect = dsharp.tensor([3.5; 1.2])
        let d = Normal(meanCorrect, stddevCorrect)
        let batchShape = d.BatchShape
        let batchShapeCorrect = [|2|]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.mean(0)
        let stddev = samples.stddev(0)
        let logprob = d.Logprob(dsharp.tensor([20.; 21.]))
        let logprobCorrect = dsharp.tensor([-6.2533; -1.4485])

        Assert.AreEqual(batchShapeCorrect, batchShape)
        Assert.AreEqual(eventShapeCorrect, eventShape)
        Assert.True(mean.allclose(meanCorrect, 0.1))
        Assert.True(stddev.allclose(stddevCorrect, 0.1))
        Assert.True(logprob.allclose(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsUniform () =
        let d = Uniform(dsharp.tensor(0.5), dsharp.tensor(10.5))
        let batchShape = d.BatchShape
        let batchShapeCorrect = [||]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]        
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.mean()
        let stddev = samples.stddev()
        let meanCorrect = dsharp.tensor(5.5)
        let stddevCorrect = dsharp.tensor(2.8868)
        let logprob = d.Logprob(dsharp.tensor(8.))
        let logprobCorrect = dsharp.tensor(-2.3026)

        Assert.AreEqual(batchShapeCorrect, batchShape)
        Assert.AreEqual(eventShapeCorrect, eventShape)
        Assert.True(mean.allclose(meanCorrect, 0.1))
        Assert.True(stddev.allclose(stddevCorrect, 0.1))
        Assert.True(logprob.allclose(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsUniformBatched () =
        let d = Uniform(dsharp.tensor([0.5; 0.; -5.]), dsharp.tensor([10.5; 1.; 5.]))
        let batchShape = d.BatchShape
        let batchShapeCorrect = [|3|]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]           
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.mean(0)
        let stddev = samples.stddev(0)
        let meanCorrect = dsharp.tensor([5.5; 0.5; 0.])
        let stddevCorrect = dsharp.tensor([2.8868; 0.28867; 2.88435])
        let logprob = d.Logprob(dsharp.tensor([8.; 0.2; 4.]))
        let logprobCorrect = dsharp.tensor([-2.3026; 0.; -2.3026])

        Assert.AreEqual(batchShapeCorrect, batchShape)
        Assert.AreEqual(eventShapeCorrect, eventShape)
        Assert.True(mean.allclose(meanCorrect, 0.1, 0.1))
        Assert.True(stddev.allclose(stddevCorrect, 0.1, 0.1))
        Assert.True(logprob.allclose(logprobCorrect, 0.1, 0.1))

