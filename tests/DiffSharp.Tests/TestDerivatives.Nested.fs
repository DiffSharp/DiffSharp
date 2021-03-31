// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Util

// #nowarn "0058"

[<TestFixture>]
type TestDerivativesNested () =

    [<Test>]
    member _.TestDerivativesNestedPerturbationConfusion () =
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        let x0 = dsharp.tensor(1)
        let y0 = dsharp.tensor(2)
        let d = dsharp.diff (fun x -> x * dsharp.diff (fun y -> x * y) y0) x0
        let dCorrect = dsharp.tensor(2)
        Assert.CheckEqual(dCorrect, d)
