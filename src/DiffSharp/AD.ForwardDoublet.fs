//
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright 2014 National University of Ireland Maynooth.
// All rights reserved.
//
// Written by:
//
//   agilim Gunes Baydin
//   agilimgunes.baydin@nuim.ie
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

//
// Reference for "doublets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Forward AD module, using doublets for gradient computation
module DiffSharp.AD.ForwardDoublet

open DiffSharp.Util
open DiffSharp.Util.General

/// DualD numeric type, keeping a doublet of primal value and a vector of gradient components
// UNOPTIMIZED
type DualD =
    | DualD of float * Vector
    override d.ToString() = let (DualD(p, g)) = d in sprintf "DualD (%f, %A)" p g
    static member op_Explicit(p) = DualD(p, Vector.Zero)
    static member op_Explicit(DualD(p, _)) = p
    static member DivideByInt(DualD(p, g), i:int) = DualD(p / float i, g / float i)
    static member Zero = DualD(0., Vector.Zero)
    static member One = DualD(1., Vector.Zero)
    static member (+) (DualD(a, ag), DualD(b, bg)) = DualD(a + b, ag + bg)
    static member (-) (DualD(a, ag), DualD(b, bg)) = DualD(a - b, ag - bg)
    static member (*) (DualD(a, ag), DualD(b, bg)) = DualD(a * b, ag * b + a * bg)
    static member (/) (DualD(a, ag), DualD(b, bg)) = DualD(a / b, (ag * b - a * bg) / (b * b))
    static member Pow (DualD(a, ag), DualD(b, bg)) = DualD(a ** b, (a ** b) * ((b * ag / a) + ((log a) * bg)))
    static member (+) (DualD(a, ag), b) = DualD(a + b, ag)
    static member (-) (DualD(a, ag), b) = DualD(a - b, ag)
    static member (*) (DualD(a, ag), b) = DualD(a * b, ag * b)
    static member (/) (DualD(a, ag), b) = DualD(a / b, ag / b)
    static member Pow (DualD(a, ag), b) = DualD(a ** b, b * (a ** (b - 1.)) * ag)
    static member (+) (a, DualD(b, bg)) = DualD(b + a, bg)
    static member (-) (a, DualD(b, bg)) = DualD(a - b, -bg)
    static member (*) (a, DualD(b, bg)) = DualD(b * a, bg * a)
    static member (/) (a, DualD(b, bg)) = DualD(a / b, -a * bg / (b * b))
    static member Pow (a, DualD(b, bg)) = DualD(a ** b, (a ** b) * (log a) * bg)
    static member Log (DualD(a, ag)) = DualD(log a, ag / a)
    static member Exp (DualD(a, ag)) = DualD(exp a, ag * exp a)
    static member Sin (DualD(a, ag)) = DualD(sin a, ag * cos a)
    static member Cos (DualD(a, ag)) = DualD(cos a, -ag * sin a)
    static member Tan (DualD(a, ag)) = DualD(tan a, ag / ((cos a) * (cos a)))
    static member (~-) (DualD(a, ag)) = DualD(-a, -ag)
    static member Sqrt (DualD(a, ag)) = DualD(sqrt a, ag / (2. * sqrt a))
    static member Sinh (DualD(a, ag)) = DualD(sinh a, ag * cosh a)
    static member Cosh (DualD(a, ag)) = DualD(cosh a, ag * sinh a)
    static member Tanh (DualD(a, ag)) = DualD(tanh a, ag / ((cosh a) * (cosh a)))
    static member Asin (DualD(a, ag)) = DualD(asin a, ag / sqrt (1. - a * a))
    static member Acos (DualD(a, ag)) = DualD(acos a, -ag / sqrt (1. - a * a))
    static member agan (DualD(a, ag)) = DualD(atan a, ag / (1. + a * a))

/// DualD operations module (automatically opened)
[<AutoOpen>]
module DualDOps =
    /// Make DualD, with primal value `p` and all gradient components 0
    let inline dualD p m = DualD(p, Vector.Create(m, 0.))
    /// Make DualD, with primal value `p` and gradient array `g`
    let inline dualDSet (p, g) = DualD(p, Vector.Create(g))
    /// Make active DualD (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the component with index `i` having value 1, and the rest of the components 0
    let inline dualDAct p m i = DualD(p, Vector.Create(m, i, 1.))
    /// Make an array of active DualD, with primal values given in array `x`. For a DualD with index _i_, the gradient is the unit vector with 1 in the _i_th place.
    let inline dualDActArray (x:float[]) = Array.init x.Length (fun i -> dualDAct x.[i] x.Length i)
    /// Get the primal value of a DualD
    let inline primal (DualD(p, _)) = p
    /// Get the gradient array of a DualD
    let inline gradient (DualD(_, g)) = g.V
    /// Get the primal value and the first gradient component of a DualD, as a tuple
    let inline tuple (DualD(p, g)) = (p, g.V.[0])
    /// Get the primal value and the gradient array of a DualD, as a tuple
    let inline tupleG (DualD(p, g)) = (p, g.V)


/// ForwardDoublet differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardDoubletOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> dualDAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        dualDActArray >> f >> tupleG

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        fun x ->
            let a = dualDActArray x |> f
            (Array.map primal a, Matrix.Create(a.Length, fun i -> gradient a.[i]).M)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x -> let (v, j) = jacobian' f x in (v, transpose j)

    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd


