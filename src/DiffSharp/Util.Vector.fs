//
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright 2014 National University of Ireland Maynooth.
// All rights reserved.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

#light

namespace DiffSharp.Util

/// Lightweight vector type for internal usage
type Vector =
    val V : float []
    new(v) = {V = v}
    member v.Dim = v.V.Length
    member v.Item i = v.V.[i]
    override v.ToString() = sprintf "Vector %A" v.V
    /// Create Vector from array `v`
    static member Create(v) = Vector(v)
    /// Create Vector with dimension `n` and a generator function `f` to compute the elements 
    static member Create(n, f) = Vector(Array.init n f)
    /// Create Vector with dimension `n` and all elements having value `v`
    static member Create(n, v) = Vector(Array.create n v)
    /// Create Vector with dimension `n`, the element with index `i` having value `v`, and the rest of the elements 0
    static member Create(n, i, v) = Vector.Create(n, (fun j -> if j = i then v else 0.))
    /// Create zero Vector
    static member Zero = Vector.Create(0, 0.)
    /// Add Vector `a` to Vector `b`
    static member (+) (a:Vector, b:Vector) = Vector.Create(a.Dim, fun i -> a.[i] + b.[i])
    /// Subtract Vector `b` from Vector `a`
    static member (-) (a:Vector, b:Vector) = Vector.Create(a.Dim, fun i -> a.[i] - b.[i])
    /// Multiply Vector `a` and Vector `b` element-wise (Hadamard product)
    static member (*) (a:Vector, b:Vector) = Vector.Create(a.Dim, fun i -> a.[i] * b.[i])
    /// Multiply Vector `a` by float `b`
    static member (*) (a:Vector, b:float) = Vector.Create(a.Dim, fun i -> a.[i] * b)
    /// Multiply Vector `b` by float `a`
    static member (*) (a:float, b:Vector) = Vector.Create(b.Dim, fun i -> a * b.[i])
    /// Divide Vector `a` by Vector `b` element-wise
    static member (/) (a:Vector, b:Vector) = Vector.Create(a.Dim, fun i -> a.[i] / b.[i])
    /// Divide Vector `a` by float `b`
    static member (/) (a:Vector, b:float) = Vector.Create(a.Dim, fun i -> a.[i] / b)
    /// Create Vector whose elements are float `a` divided by the corresponding element of Vector `b`
    static member (/) (a:float, b:Vector) = Vector.Create(b.Dim, fun i -> a / b.[i])
    /// Negative of Vector `a`
    static member (~-) (a:Vector) = Vector.Create(a.Dim, fun i -> -a.[i])
    