// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp

// #nowarn "0058"

[<TestFixture>]
module TestOps =

    type Tensor with
        member a.sinExt() = 
            Tensor.Op
                { new UnaryOpElementwise() with 
                    member _.fRaw(a) = a.SinT()
                    member _.df_da(a,f) = a.cosExt()
                }
                (a)

        member a.cosExt() =
            Tensor.Op
                { new UnaryOpElementwise() with
                    member _.fRaw(a) = a.CosT()
                    member _.df_da(a,f) = -a.sinExt()
                }
                (a)

        member a.expExt() =
            Tensor.Op
                { new UnaryOpElementwise() with
                    member _.fRaw(a) = a.ExpT()
                    member _.df_da(a,f) = f
                }
                (a)

        member a.logExt() =
            Tensor.Op
                { new UnaryOpElementwise() with
                    member _.fRaw(a) = a.LogT()
                    member _.df_da(a,f) = 1/a
                }
                (a)

        member a.transposeExt() =
            Tensor.Op
                { new UnaryOp() with
                    member _.fRaw(a) = a.TransposeT2()
                    member _.ad_df_da(a,ad,f) = ad.transposeExt()
                    member _.fd_df_da(a,f,fd) = fd.transposeExt()
                }
                (a)

        member a.powExt(b) =
            Tensor.Op
                { new BinaryOpElementwise() with
                    member _.fRaw(a,b) = a.PowTT(b)
                    member _.df_da(a,b,f) = b * f / a
                    member _.df_db(a,b,f) = f * a.logExt()
                }
                (a,b)

        member a.mulExt(b) =
            Tensor.Op
                { new BinaryOpElementwise() with
                    member _.fRaw(a,b) = a.MulTT(b)
                    member _.df_da(a,b,f) = b
                    member _.df_db(a,b,f) = a
                }
                (a,b)

        member a.matmulExt(b) =
            Tensor.Op
                { new BinaryOp() with
                    member _.fRaw(a,b) = a.MatMulTT(b)
                    member _.ad_df_da(a,ad,b,f) = ad.matmulExt(b)
                    member _.bd_df_db(a,b,bd,f) = a.matmulExt(bd)
                    member _.fd_df_da(a,b,f,fd) = fd.matmulExt(b.transposeExt())
                    member _.fd_df_db(a,b,f,fd) = a.transposeExt().matmulExt(fd)
                }
                (a,b)

    let compareUnaryOps op1 op2 count initializer =
        for i in 1..count do
            let x = initializer()
            let xd = dsharp.randnLike(x)
            let zd = dsharp.randnLike(x)

            let fwdx = x.forwardDiff(xd)
            let fwdz1 : Tensor = op1 fwdx
            let fwdz2 : Tensor = op2 fwdx
            let fwdzd1 = fwdz1.derivative
            let fwdzd2 = fwdz2.derivative

            let revx1 = x.reverseDiff()
            let revx2 = x.reverseDiff()
            let revz1 = op1 revx1
            let revz2 = op1 revx2
            revz1.reverse(zd)
            revz2.reverse(zd)
            let revxd1 = revx1.derivative
            let revxd2 = revx2.derivative

            printfn "fwdz1 %A" fwdz1
            printfn "fwdz2 %A" fwdz2
            printfn "fwdzd1 %A" fwdzd1
            printfn "fwdzd2 %A" fwdzd2
            printfn "revz1 %A" revz1
            printfn "revz2 %A" revz2
            printfn "revxd1 %A" revxd1
            printfn "revxd2 %A\n" revxd2

            Assert.True(fwdz1.allclose(fwdz2, 0.01))
            Assert.True(fwdzd1.allclose(fwdzd2, 0.01))
            Assert.True(revz1.allclose(revz2, 0.01))
            Assert.True(revxd1.allclose(revxd2, 0.01))

    [<Test>]
    let TestExtensions() =
        compareUnaryOps (fun t -> t.sin()) (fun t -> t.sinExt()) 10 (fun () -> dsharp.randn(10))
        compareUnaryOps (fun t -> t.cos()) (fun t -> t.cosExt()) 10 (fun () -> dsharp.randn(10))
        compareUnaryOps (fun t -> t.exp()) (fun t -> t.expExt()) 10 (fun () -> dsharp.randn(10))
        compareUnaryOps (fun t -> t.log()) (fun t -> t.logExt()) 10 (fun () -> dsharp.randn(10).abs())