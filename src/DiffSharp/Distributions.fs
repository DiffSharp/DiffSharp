namespace DiffSharp.Distributions
open DiffSharp
open DiffSharp.Util

[<AbstractClass>]
type Distribution(name:string) =
    let name = name
    abstract member Sample: unit -> Tensor

type Normal(mean:Tensor, stddev:Tensor) =
    inherit Distribution("Normal")

    member d.Mean = mean
    member d.Stddev = stddev

    override d.Sample() =
        // let meanValue = float d.Mean
        // let stddevValue = float d.Stddev
        d.Mean + Random.Normal() * d.Stddev