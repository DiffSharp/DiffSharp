// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

[<AutoOpen>]
module OpNormExtensions =

    type Tensor with
        member a.norm(?order:float, ?dim:int, ?keepDim:bool) =
            if not (a.dtype = Dtype.Float32 || a.dtype = Dtype.Float64) then failwithf "Vector norm is only supported for Float32 and Float64 dtypes."
            let order = defaultArg order 2.
            match order, dim with
            | 1., None -> a.flatten().abs().sum()
            | 1., Some(dim) -> a.abs().sum(dim=dim, ?keepDim=keepDim)
            | 2., None -> let aa = a.flatten() in (aa * aa).sum().sqrt()
            | 2., Some(dim) -> (a * a).sum(dim=dim, ?keepDim=keepDim).sqrt()
            | System.Double.PositiveInfinity, None -> a.flatten().abs().max()
            | System.Double.PositiveInfinity, Some(dim) -> a.abs().max(dim=dim, ?keepDim=keepDim)
            | System.Double.NegativeInfinity, None -> a.flatten().abs().min()
            | System.Double.NegativeInfinity, Some(dim) -> a.abs().min(dim=dim, ?keepDim=keepDim)
            | 0., None -> a.ne(a.zerosLike()).cast(dtype=a.dtype).sum()
            | 0., Some(dim) -> a.ne(a.zerosLike()).cast(dtype=a.dtype).sum(dim=dim, ?keepDim=keepDim)
            | order, None -> a.abs().pow(order).sum().pow(1./order)
            | order, Some(dim) -> a.abs().pow(order).sum(dim=dim, ?keepDim=keepDim).pow(1./order)

    type dsharp with
        static member norm(a:Tensor, ?order:float, ?dim:int, ?keepDim:bool) = a.norm(?order=order, ?dim=dim, ?keepDim=keepDim)
