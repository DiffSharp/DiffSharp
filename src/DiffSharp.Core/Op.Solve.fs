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
                    member _.fd_dfda(a,b,f,fd) = a.zerosLike() // let ba = a.transpose(-2, -1).solve(fd) in ba// let ba = a.transpose(-2, -1).solve(fd) in -ba.outer(fd)
                    member _.fd_dfdb(a,b,f,fd) = b.zerosLike() // let ba = a.transpose(-2, -1).solve(fd) in ba
                }
                (a,b)

    type dsharp with
        static member solve(a:Tensor, b:Tensor) = a.solve(b)
