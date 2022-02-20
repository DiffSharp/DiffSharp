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
                    member _.ad_dfda(a:Tensor,ad,f) = 
                        if a.dim = 2 then
                            // The following differs from Jacobi's formula which has a trace instead of a sum
                            // But it is confirmed to be correct by reverse-mode-based forward-mode eval and also finite differences
                            f * (a.inv().transpose() * ad).sum()
                        else
                            f * (a.inv().transpose(-1, -2) * ad).flatten(1).sum(-1)
                    member _.fd_dfda(a,f,fd) = 
                        if a.dim = 2 then 
                            fd * f * a.inv().transpose()
                        else
                            // Ugly but correct
                            fd.unsqueeze(1).unsqueeze(1) * f.unsqueeze(1).unsqueeze(1) * a.inv().transpose(-1, -2)
                }
                (a)

    type dsharp with
        static member det(a:Tensor) = a.det()
