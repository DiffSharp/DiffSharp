module Symbolic

open FsCheck.NUnit
open DiffSharp.Util
open DiffSharp.Tests
open NUnit.Framework

module Float64 = 
    open DiffSharp.Symbolic.Float64
    [<ReflectedDefinition>]
    let f2 x = x**2.0

    [<Test>]
    // See https://github.com/DiffSharp/DiffSharp/issues/27
    let ``Diff.Of.Exponential.At.Zero``() = 
        let dy = diff <@ f2 @>
        Assert.True( Util.(=~) (dy 0.0, 0.0))

    [<Test>]
    let ``Diff.Of.Exponential.At.One``() = 
        let dy = diff <@ f2 @>
        Assert.True(Util.(=~) (dy 1.0, 2.0))

module Float32 = 
    open DiffSharp.Symbolic.Float32

    [<ReflectedDefinition>]
    let f2 x = x**2.0f

    [<Test>]
    // See https://github.com/DiffSharp/DiffSharp/issues/27
    let ``Diff.Of.Exponential.At.Zero``() = 
        printfn "testing"
        let dy = diff <@ f2 @>
        Assert.True(Util.(=~) (dy 0.0f, 0.0f))

    [<Test>]
    let ``Diff.Of.Exponential.At.One``() = 
        let dy = diff <@ f2 @>
        Assert.True(Util.(=~) (dy 1.0f, 2.0f))

