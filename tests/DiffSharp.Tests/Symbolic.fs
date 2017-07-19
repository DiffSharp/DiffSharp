module Symbolic

open FsCheck.NUnit
open DiffSharp.Util
open DiffSharp.Tests

module Float64 = 
    open DiffSharp.Symbolic.Float64
    [<ReflectedDefinition>]
    let f2 x = x**2.0

    [<Property>]
    // See https://github.com/DiffSharp/DiffSharp/issues/27
    let ``Diff.Of.Exponential.At.Zero``() = 
        let dy = diff <@ f2 @>
        Util.(=~) (dy 0.0, 0.0)

    [<Property>]
    let ``Diff.Of.Exponential.At.One``() = 
        let dy = diff <@ f2 @>
        Util.(=~) (dy 1.0, 2.0)

module Float32 = 
    open DiffSharp.Symbolic.Float32
    [<ReflectedDefinition>]
    let f2 x = x**2.0f

    [<Property>]
    // See https://github.com/DiffSharp/DiffSharp/issues/27
    let ``Diff.Of.Exponential.At.Zero``() = 
        let dy = diff <@ f2 @>
        Util.(=~) (dy 0.0f, 0.0f)

    [<Property>]
    let ``Diff.Of.Exponential.At.One``() = 
        let dy = diff <@ f2 @>
        Util.(=~) (dy 1.0f, 2.0f)

