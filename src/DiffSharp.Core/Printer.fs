// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

type Printer =
    | Default
    | Short
    | Full
    | Custom of threshold: int * edgeItems: int * precision: int

    member p.threshold =
        match p with
        | Default -> 100
        | Short -> 10
        | Full -> System.Int32.MaxValue
        | Custom(t, _, _) -> t

    member p.edgeItems =
        match p with
        | Default -> 3
        | Short -> 2
        | Full -> -1
        | Custom(_, e, _) -> e

    member p.precision =
        match p with
        | Default -> 4
        | Short -> 2
        | Full -> 4
        | Custom(_, _, p) -> p

/// Contains functions and settings related to print options.
module Printer = 

    /// Get or set the default printer used when printing tensors. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default : Printer = Printer.Default