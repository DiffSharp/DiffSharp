// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpBMMExtensions =

    type Tensor with
        /// <summary>Batched matrix product of two tensors. Tensors <paramref name="b" /> must be 3d tensors each containing the same number of matrices. If the tensor is a \(b \times n \times m\) tensor, and <paramref name="b" /> is a \(b \times m \times p\) tensor, the result will be a \(b \times n \times p\) tensor.</summary>
        /// <param name="b">The second tensor.</param>
        member a.bmm(b:Tensor) =
            Shape.checkCanBMM a.shape b.shape |> ignore
            Tensor.Op
                { new BinaryOp("bmm") with 
                    member _.fRaw(a,b) = a.BMMTT(b)
                    member _.ad_dfda(a,ad,b,f) = ad.bmm(b)
                    member _.bd_dfdb(a,b,bd,f) = a.bmm(bd)
                    member _.fd_dfda(a,b,f,fd) = fd.bmm(b.transpose(1, 2))
                    member _.fd_dfdb(a,b,f,fd) = a.transpose(1, 2).bmm(fd)
                }
                (a,b)

    type dsharp with
        /// <summary>Batched matrix product of two tensors. Tensors <paramref name="a" /> and  <paramref name="b" /> must be 3d tensors each containing the same number of matrices. If <paramref name="a" /> is a \(b \times n \times m\) tensor, <paramref name="b" /> is a \(b \times m \times p\) tensor, the result will be a \(b \times n \times p\) tensor.</summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        static member bmm(a:Tensor, b:Tensor) = a.bmm(b)
