// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

type Func(func:Tensor->Tensor, ?name:string) =
    inherit Model()
    // TODO: we could get and print the func name using reflection, quotations etc.
    override _.ToString() = sprintf "Func(%s)" (if name.IsSome then name.Value else "")
    override _.forward(x) = func(x)