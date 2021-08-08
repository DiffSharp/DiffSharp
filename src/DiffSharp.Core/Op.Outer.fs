// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpOuterExtensions =

    type Tensor with
        member a.outer(b:Tensor) =
            match a.dim, b.dim with
            | 1, 1 -> a.unsqueeze(1).matmul(b.unsqueeze(0))
            | 2, 2 -> a.unsqueeze(2).bmm(b.unsqueeze(1))
            | _ -> failwithf "Outer product unsupported for tensor shapes %A %A" a.shape b.shape

    type dsharp with
        static member outer(a:Tensor, b:Tensor) = a.outer(b)
