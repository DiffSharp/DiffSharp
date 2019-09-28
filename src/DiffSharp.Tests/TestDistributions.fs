namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Distributions

[<TestFixture>]
type TestDistributions () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestDistributionsNormal () =

        let meanCorrect = Tensor.Create(10.)
        let stddevCorrect = Tensor.Create(3.5)
        let d = Normal(meanCorrect, stddevCorrect)
        let samples = d.Sample(5000)
        let mean = samples.Mean()
        let stddev = samples.Stddev()
        let logprob = d.Logprob(Tensor.Create(20.))
        let logprobCorrect = Tensor.Create(-6.2533)

        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))

    [<Test>]
    member this.TestDistributionsUniform () =

        let d = Uniform(Tensor.Create(0.5), Tensor.Create(10.5))
        let samples = d.Sample(5000)
        let mean = samples.Mean()
        let stddev = samples.Stddev()
        let meanCorrect = Tensor.Create(5.5)
        let stddevCorrect = Tensor.Create(2.8868)
        let logprob = d.Logprob(Tensor.Create(8.))
        let logprobCorrect = Tensor.Create(-2.3026)

        Assert.True(mean.ApproximatelyEqual(meanCorrect, 0.1))
        Assert.True(stddev.ApproximatelyEqual(stddevCorrect, 0.1))
        Assert.True(logprob.ApproximatelyEqual(logprobCorrect, 0.1))

