namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =
    let numEmpiricalSamples = 5000

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestDistributionsNormal () =

        let meanCorrect = Tensor.Create(10.)
        let stddevCorrect = Tensor.Create(3.5)
        let d = Normal(meanCorrect, stddevCorrect)
        let batchShape = d.BatchShape
        let batchShapeCorrect = [||]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.Mean()
        let stddev = samples.Stddev()
        let logprob = d.Logprob(Tensor.Create(20.))
        let logprobCorrect = Tensor.Create(-6.2533)

        Assert.AreEqual(batchShape, batchShapeCorrect)
        Assert.AreEqual(eventShape, eventShapeCorrect)
        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsNormalBatched () =

        let meanCorrect = Tensor.Create([10.; 20.])
        let stddevCorrect = Tensor.Create([3.5; 1.2])
        let d = Normal(meanCorrect, stddevCorrect)
        let batchShape = d.BatchShape
        let batchShapeCorrect = [|2|]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.Mean(0)
        let stddev = samples.Stddev(0)
        let logprob = d.Logprob(Tensor.Create([20.; 21.]))
        let logprobCorrect = Tensor.Create([-6.2533; -1.4485])

        Assert.AreEqual(batchShape, batchShapeCorrect)
        Assert.AreEqual(eventShape, eventShapeCorrect)
        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsUniform () =

        let d = Uniform(Tensor.Create(0.5), Tensor.Create(10.5))
        let batchShape = d.BatchShape
        let batchShapeCorrect = [||]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]        
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.Mean()
        let stddev = samples.Stddev()
        let meanCorrect = Tensor.Create(5.5)
        let stddevCorrect = Tensor.Create(2.8868)
        let logprob = d.Logprob(Tensor.Create(8.))
        let logprobCorrect = Tensor.Create(-2.3026)

        Assert.AreEqual(batchShape, batchShapeCorrect)
        Assert.AreEqual(eventShape, eventShapeCorrect)
        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))


    [<Test>]
    member this.TestDistributionsUniformBatched () =

        let d = Uniform(Tensor.Create([0.5; 0.; -5.]), Tensor.Create([10.5; 1.; 5.]))
        let batchShape = d.BatchShape
        let batchShapeCorrect = [|3|]
        let eventShape = d.EventShape
        let eventShapeCorrect = [||]           
        let samples = d.Sample(numEmpiricalSamples)
        let mean = samples.Mean(0)
        let stddev = samples.Stddev(0)
        let meanCorrect = Tensor.Create([5.5; 0.5; 0.])
        let stddevCorrect = Tensor.Create([2.8868; 0.28867; 2.88435])
        let logprob = d.Logprob(Tensor.Create([8.; 0.2; 4.]))
        let logprobCorrect = Tensor.Create([-2.3026; 0.; -2.3026])

        Assert.AreEqual(batchShape, batchShapeCorrect)
        Assert.AreEqual(eventShape, eventShapeCorrect)
        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))

