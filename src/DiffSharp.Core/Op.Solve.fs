// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpSolveExtensions =

    type Tensor with
        member a.solve(b:Tensor) =
            let _ = Shape.checkCanSolve a.shape b.shape
            Tensor.Op
                { new BinaryOp("solve") with
                    member _.fRaw(a,b) = a.SolveTT(b)
                    member _.ad_dfda(a,ad,b,f) = 
                        match a.dim, b.dim with
                        | 3, 2 -> let aa:Tensor = a.solve(-ad.matmul(f.unsqueeze(-1))) in aa.squeeze(-1)
                        | _ -> a.solve(-ad.matmul(f))
                    member _.bd_dfdb(a,b,bd,f) = a.solve(bd)
                    member _.fd_dfda(a,b,f,fd) = 
                        let ba = a.transpose(-2, -1).solve(fd)
                        match a.dim, b.dim with
                        | 2, 1 -> -ba.outer(f)
                        | 3, 2 -> -ba.unsqueeze(-1).matmul(f.unsqueeze(-1).transpose(-2, -1))
                        | _ -> -ba.matmul(f.transpose(-2, -1))
                    member _.fd_dfdb(a,b,f,fd) = a.transpose(-2, -1).solve(fd)
                }
                (a,b)

    type dsharp with
        static member solve(a:Tensor, b:Tensor) = a.solve(b)
