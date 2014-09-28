//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
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
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Reference for "doublets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Forward AD module, using a vector of gradient components
module DiffSharp.AD.ForwardV

open DiffSharp.Util
open DiffSharp.Util.General

/// DualV numeric type, keeping a doublet of primal value and a vector of gradient components
// UNOPTIMIZED
type DualV =
    | DualV of float * Vector
    override d.ToString() = let (DualV(p, g)) = d in sprintf "DualV (%f, %A)" p g
    static member op_Explicit(p) = DualV(p, Vector.Zero)
    static member op_Explicit(DualV(p, _)) = p
    static member DivideByInt(DualV(p, g), i:int) = DualV(p / float i, g / float i)
    static member Zero = DualV(0., Vector.Zero)
    static member One = DualV(1., Vector.Zero)
    static member (+) (DualV(a, ag), DualV(b, bg)) = DualV(a + b, ag + bg)
    static member (-) (DualV(a, ag), DualV(b, bg)) = DualV(a - b, ag - bg)
    static member (*) (DualV(a, ag), DualV(b, bg)) = DualV(a * b, ag * b + a * bg)
    static member (/) (DualV(a, ag), DualV(b, bg)) = DualV(a / b, (ag * b - a * bg) / (b * b))
    static member Pow (DualV(a, ag), DualV(b, bg)) = DualV(a ** b, (a ** b) * ((b * ag / a) + ((log a) * bg)))
    static member (+) (DualV(a, ag), b) = DualV(a + b, ag)
    static member (-) (DualV(a, ag), b) = DualV(a - b, ag)
    static member (*) (DualV(a, ag), b) = DualV(a * b, ag * b)
    static member (/) (DualV(a, ag), b) = DualV(a / b, ag / b)
    static member Pow (DualV(a, ag), b) = DualV(a ** b, b * (a ** (b - 1.)) * ag)
    static member (+) (a, DualV(b, bg)) = DualV(b + a, bg)
    static member (-) (a, DualV(b, bg)) = DualV(a - b, -bg)
    static member (*) (a, DualV(b, bg)) = DualV(b * a, bg * a)
    static member (/) (a, DualV(b, bg)) = DualV(a / b, -a * bg / (b * b))
    static member Pow (a, DualV(b, bg)) = DualV(a ** b, (a ** b) * (log a) * bg)
    static member Log (DualV(a, ag)) = DualV(log a, ag / a)
    static member Exp (DualV(a, ag)) = DualV(exp a, ag * exp a)
    static member Sin (DualV(a, ag)) = DualV(sin a, ag * cos a)
    static member Cos (DualV(a, ag)) = DualV(cos a, -ag * sin a)
    static member Tan (DualV(a, ag)) = DualV(tan a, ag / ((cos a) * (cos a)))
    static member (~-) (DualV(a, ag)) = DualV(-a, -ag)
    static member Sqrt (DualV(a, ag)) = DualV(sqrt a, ag / (2. * sqrt a))
    static member Sinh (DualV(a, ag)) = DualV(sinh a, ag * cosh a)
    static member Cosh (DualV(a, ag)) = DualV(cosh a, ag * sinh a)
    static member Tanh (DualV(a, ag)) = DualV(tanh a, ag / ((cosh a) * (cosh a)))
    static member Asin (DualV(a, ag)) = DualV(asin a, ag / sqrt (1. - a * a))
    static member Acos (DualV(a, ag)) = DualV(acos a, -ag / sqrt (1. - a * a))
    static member agan (DualV(a, ag)) = DualV(atan a, ag / (1. + a * a))

/// DualV operations module (automatically opened)
[<AutoOpen>]
module DualVOps =
    /// Make DualV, with primal value `p` and all gradient components 0
    let inline dualV p m = DualV(p, Vector.Create(m, 0.))
    /// Make DualV, with primal value `p` and gradient array `g`
    let inline dualVSet (p, g) = DualV(p, Vector.Create(g))
    /// Make active DualV (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the component with index `i` having value 1, and the rest of the components 0
    let inline dualVAct p m i = DualV(p, Vector.Create(m, i, 1.))
    /// Make an array of active DualV, with primal values given in array `x`. For a DualV with index _i_, the gradient is the unit vector with 1 in the _i_th place.
    let inline dualVActArray (x:float[]) = Array.init x.Length (fun i -> dualVAct x.[i] x.Length i)
    /// Get the primal value of a DualV
    let inline primal (DualV(p, _)) = p
    /// Get the gradient array of a DualV
    let inline gradient (DualV(_, g)) = g.V
    /// Get the primal value and the first gradient component of a DualV, as a tuple
    let inline tuple (DualV(p, g)) = (p, g.V.[0])
    /// Get the primal value and the gradient array of a DualV, as a tuple
    let inline tupleG (DualV(p, g)) = (p, g.V)


/// ForwardV differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardVOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> dualVAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        dualVActArray >> f >> tupleG

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        fun x ->
            let a = dualVActArray x |> f
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


