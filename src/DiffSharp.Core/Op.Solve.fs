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
                    member _.ad_dfda(a,ad,b,f) = a.solve(-ad.matmul(f))
                    member _.bd_dfdb(a,b,bd,f) = a.solve(bd)
                    member _.fd_dfda(a,b,f,fd) = 
                        let ba = a.transpose(-2, -1).solve(fd)
                        match a.dim, b.dim with
                        | 2, 2 -> -ba.matmul(f.transpose(-2, -1))
                        | 2, 1 -> -ba.outer(f)
                        | 3, 3 -> -ba.matmul(f.transpose(-2, -1))
                        | 3, 2 -> -ba.bmm(f.unsqueeze(-1).transpose(-2, -1))
                        | _ -> failwithf "Solve reverse mode unexpected case, a.dim %A and b.dim %A" a.dim b.dim
                    member _.fd_dfdb(a,b,f,fd) = a.transpose(-2, -1).solve(fd)
                }
                (a,b)

    type dsharp with
        static member solve(a:Tensor, b:Tensor) = a.solve(b)
