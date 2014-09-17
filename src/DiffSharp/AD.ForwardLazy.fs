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

/// Forward AD module, lazy higher-order
module DiffSharp.AD.ForwardLazy

open DiffSharp.Util
open DiffSharp.Util.General

/// DualL numeric type, where the tangent value is another DualL, forming a lazy chain of higher-order derivatives
// UNOPTIMIZED
type DualL =
    | DualL of float * Lazy<DualL>
    static member Create(p) = DualL(p, Lazy<DualL>(fun () -> DualL.Zero))
    static member Create(p, t) = DualL(p, Lazy<DualL>(fun () -> DualL.Create(t)))
    static member op_Explicit(p) = DualL.Create(p)
    static member op_Explicit(DualL(p, _)) = p
    override d.ToString() = let (DualL(p, t)) = d in sprintf "DualL (%f, %f)" p (float t.Value)
    member d.P = let (DualL(p, _)) = d in p
    member d.T = let (DualL(_, t)) = d in t.Value
    static member Zero = DualL(0., Lazy<DualL>(fun () -> DualL.Zero))
    static member One = DualL(1., Lazy<DualL>(fun () -> DualL.Zero))
    static member (+) (a:DualL, b:DualL) = DualL(a.P + b.P, Lazy<DualL>(fun () -> a.T + b.T))
    static member (-) (a:DualL, b:DualL) = DualL(a.P - b.P, Lazy<DualL>(fun () -> a.T - b.T))
    static member (*) (a:DualL, b:DualL) = DualL(a.P * b.P, Lazy<DualL>(fun () -> a.T * b + a * b.T))
    static member (/) (a:DualL, b:DualL) = DualL(a.P / b.P, Lazy<DualL>(fun () -> (a.T * b - a * b.T) / (b * b)))
    static member Pow (a:DualL, b:DualL) = DualL(a.P ** b.P, Lazy<DualL>(fun () -> (a ** b) * ((b * a.T / a) + ((log a) * b.T))))
    static member (+) (a:DualL, b) = DualL(a.P + b, Lazy<DualL>(fun () -> a.T))
    static member (-) (a:DualL, b) = DualL(a.P - b, Lazy<DualL>(fun () -> a.T))
    static member (*) (a:DualL, b) = DualL(a.P * b, Lazy<DualL>(fun () -> a.T * b))
    static member (/) (a:DualL, b) = DualL(a.P / b, Lazy<DualL>(fun () -> a.T / b))
    static member Pow (a:DualL, b) = DualL(a.P ** b, Lazy<DualL>(fun () -> b * (a ** (b - 1.)) * a.T))
    static member (+) (a, b:DualL) = DualL(b.P + a, Lazy<DualL>(fun () -> b.T))
    static member (-) (a, b:DualL) = DualL(a - b.P, Lazy<DualL>(fun () -> -b.T))
    static member (*) (a, b:DualL) = DualL(b.P * a, Lazy<DualL>(fun () -> b.T * a))
    static member (/) (a, b:DualL) = DualL(a / b.P, Lazy<DualL>(fun () -> -a * b.T / (b * b)))
    static member Pow (a, b:DualL) = DualL(a ** b.P, Lazy<DualL>(fun () -> (DualL.Create(a) ** b) * (log a) * b.T))
    static member Log (a:DualL) = DualL(log a.P, Lazy<DualL>(fun () -> a.T / a))
    static member Exp (a:DualL) = DualL(exp a.P, Lazy<DualL>(fun () -> a.T * exp a))
    static member Sin (a:DualL) = DualL(sin a.P, Lazy<DualL>(fun () -> a.T * cos a))
    static member Cos (a:DualL) = DualL(cos a.P, Lazy<DualL>(fun () -> -a.T * sin a))
    static member Tan (a:DualL) = DualL(tan a.P, Lazy<DualL>(fun () -> a.T / ((cos a) * (cos a))))
    static member (~-) (a:DualL) = DualL(-a.P, Lazy<DualL>(fun () -> -a.T))
    static member Sqrt (a:DualL) = DualL(sqrt a.P, Lazy<DualL>(fun () -> a.T / (2. * sqrt a)))
    static member Sinh (a:DualL) = DualL(sinh a.P, Lazy<DualL>(fun () -> a.T * cosh a))
    static member Cosh (a:DualL) = DualL(cosh a.P, Lazy<DualL>(fun () -> a.T * sinh a))
    static member Tanh (a:DualL) = DualL(tanh a.P, Lazy<DualL>(fun () -> a.T / ((cosh a) * (cosh a))))
    static member Asin (a:DualL) = DualL(asin a.P, Lazy<DualL>(fun () -> a.T / sqrt (1. - a * a)))
    static member Acos (a:DualL) = DualL(acos a.P, Lazy<DualL>(fun () -> -a.T / sqrt (1. - a * a)))
    static member Atan (a:DualL) = DualL(atan a.P, Lazy<DualL>(fun () -> a.T / (1. + a * a)))

/// DualL operations module (automatically opened)
[<AutoOpen>]
module DualLOps =
    /// Make DualL, with primal value `p` and tangent 0
    let inline dualL p = DualL.Create(p, 0.)
    /// Make DualL, with primal value `p` and tangent value `t`
    let inline dualLSet (p, t) = DualL.Create(p, t)
    /// Make active DualL (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualLAct p = DualL.Create(p, 1.)
    /// Make an array of arrays of DualL, with primal values given in array `x`. The tangent values along the diagonal are 1, the rest are 0.
    let inline dualLActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dualLAct x.[j] else dualL x.[j])))
    /// Get the primal value of a DualL
    let inline primal (DualL(p, _)) = p
    /// Get the tangent value of a DualL
    let inline tangent (DualL(_, t)) = float t.Value
    /// Get the tangent-of-tangent value of a DualL
    let inline tangent2 (d:DualL) = tangent d.T
    /// Get the primal and tangent value of a DualL, as a tuple
    let inline tuple (DualL(p, t)) = (p, float t.Value)

    /// Compute the `n`-th derivative of a DualL
    let rec diffLazy n =
        match n with
        | a when a < 0 -> failwith "Order of derivative cannot be negative."
        | 0 -> fun (x:DualL) -> x
        | 1 -> fun x -> x.T
        | _ -> fun x -> diffLazy (n - 1) x.T

    /// Custom operator (/^) for differentiation. Usage: `x` /^ `n`, value of the `n`-th order derivative of `x`.
    let ( /^ ) x n =
        diffLazy n x |> primal


/// ForwardLazy differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardLazyOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        dualLAct >> f >> tuple
    
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f =
        dualLAct >> f >> fun a -> (primal a, tangent2 a)
        
    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f =
        diff2' f >> snd

    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn' n f =
        dualLAct >> f >> diffLazy n >> tuple

    /// `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn n f =
        diffn' n f >> snd

    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f =
        fun x -> 
            Array.zip x r
            |> Array.map dualLSet
            |> f 
            |> tuple

    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r f =
        diffdir' r f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        fun x ->
            let a = Array.map f (dualLActArrayArray x)
            (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
            
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x ->
            let a = Array.map f (dualLActArrayArray x)
            (let (DualL(p, _)) = a.[0] in p, Array.sumBy tangent2 a)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x ->
            let a = Array.map f (dualLActArrayArray x)
            (Array.map primal a.[0], Array2D.map tangent (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd

    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        jacobianT' f >> fun (v, j) -> (v, transpose j)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd
