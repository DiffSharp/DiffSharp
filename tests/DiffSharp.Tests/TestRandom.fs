// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp.Util

[<TestFixture>]
type TestRandom () =

    [<Test>]
    member _.TestRandomSeed () =
        Random.Seed(1)
        let a1 = Random.Uniform()
        Random.Seed(1)
        let a2 = Random.Uniform()
        let a3 = Random.Uniform()

        Assert.AreEqual(a1, a2)
        Assert.AreNotEqual(a2, a3)

    [<Test>]
    member _.TestRandomUUID () =
        Random.Seed(1)
        let a1 = Random.UUID()
        Random.Seed(1)
        let a2 = Random.UUID()
        let a3 = Random.UUID()

        Assert.AreEqual(a1, a2)
        Assert.AreNotEqual(a2, a3)
