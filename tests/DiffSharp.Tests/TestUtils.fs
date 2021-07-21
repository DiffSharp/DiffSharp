// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open DiffSharp
open NUnit.Framework

[<AutoOpen>]
module TestUtils =
    let isException f = Assert.Throws<Exception>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isInvalidOp f = Assert.Throws<InvalidOperationException>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isAnyException f = Assert.Catch(TestDelegate(fun () -> f() |> ignore)) |> ignore

    type Assert with 

        /// Like Assert.AreEqual bute requires that the actual and expected are the same type
        static member CheckEqual (expected: 'T, actual: 'T) = Assert.AreEqual(box expected, box actual)

    type dsharp with
        /// <summary>Locally use the given default configuration, returning an IDisposable to revert to the previous configuration.</summary>
        /// <param name="dtype">The new default element type.</param>
        /// <param name="device">The new default device.</param>
        /// <param name="backend">The new default backend.</param>
        static member useConfig(?dtype: Dtype, ?device: Device, ?backend: Backend) = 
            let prevConfig = dsharp.config()
            dsharp.config(?dtype=dtype, ?device=device, ?backend=backend)
            { new System.IDisposable with member _.Dispose() = dsharp.config(prevConfig) }

