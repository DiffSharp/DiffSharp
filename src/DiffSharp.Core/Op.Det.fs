// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpDetExtensions =

    type Tensor with
        member a.det() =
            Shape.checkCanDet a.shape
            Tensor.Op
                { new UnaryOp("det") with 
                    member _.fRaw(a) = a.DetT()
                    member _.ad_dfda(a:Tensor,ad,f) = f * dsharp.trace(a.inv() * ad)
                    member _.fd_dfda(a,f,fd) = fd * f * a.inv().transpose()
                }
                (a)

    type dsharp with
        static member inv(a:Tensor) = a.det()
