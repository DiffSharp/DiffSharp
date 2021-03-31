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