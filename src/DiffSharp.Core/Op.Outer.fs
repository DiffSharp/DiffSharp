// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpOuterExtensions =

    type Tensor with
        /// <summary>Outer product of two tensors.</summary>
        /// <param name="b">The second tensor.</param>
        member a.outer(b:Tensor) =
            match a.dim, b.dim with
            | 1, 1 -> a.unsqueeze(1).matmul(b.unsqueeze(0))
            | 2, 2 when a.shape[0] = b.shape[0] -> a.unsqueeze(2).bmm(b.unsqueeze(1))  // Batched outer product
            | _ -> failwithf "Outer product unsupported for tensor shapes %A %A" a.shape b.shape

    type dsharp with
        /// <summary>Outer product of two tensors.</summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        static member outer(a:Tensor, b:Tensor) = a.outer(b)
