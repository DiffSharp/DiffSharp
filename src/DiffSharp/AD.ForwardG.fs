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

/// Forward AD module, keeping vectors of gradient components
module DiffSharp.AD.ForwardG

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualG numeric type, keeping a doublet of primal value and a vector of gradient components
// UNOPTIMIZED
[<CustomEquality; CustomComparison>]
type DualG =
    // Primal, vector of gradient components
    | DualG of float * Vector
    override d.ToString() = let (DualG(p, g)) = d in sprintf "DualG (%f, %A)" p g
    static member op_Explicit(p) = DualG(p, ZeroVector)
    static member op_Explicit(DualG(p, _)) = p
    static member DivideByInt(DualG(p, g), i:int) = DualG(p / float i, g / float i)
    static member Zero = DualG(0., ZeroVector)
    static member One = DualG(1., ZeroVector)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualG as d2 -> let DualG(a, _), DualG(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualG with another type of object."
    override d.Equals(other) = 
        match other with
        | :? DualG as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualG(a, b)) = d in hash [|a; b|]
    // DualG - DualG binary operations
    static member (+) (DualG(a, ag), DualG(b, bg)) = DualG(a + b, ag + bg)
    static member (-) (DualG(a, ag), DualG(b, bg)) = DualG(a - b, ag - bg)
    static member (*) (DualG(a, ag), DualG(b, bg)) = DualG(a * b, ag * b + a * bg)
    static member (/) (DualG(a, ag), DualG(b, bg)) = DualG(a / b, (ag * b - a * bg) / (b * b))
    static member Pow (DualG(a, ag), DualG(b, bg)) = let apowb = a ** b in DualG(apowb, apowb * ((b * ag / a) + ((log a) * bg)))
    // DualG - float binary operations
    static member (+) (DualG(a, ag), b) = DualG(a + b, ag)
    static member (-) (DualG(a, ag), b) = DualG(a - b, ag)
    static member (*) (DualG(a, ag), b) = DualG(a * b, ag * b)
    static member (/) (DualG(a, ag), b) = DualG(a / b, ag / b)
    static member Pow (DualG(a, ag), b) = DualG(a ** b, b * (a ** (b - 1.)) * ag)
    // float - DualG binary operations
    static member (+) (a, DualG(b, bg)) = DualG(b + a, bg)
    static member (-) (a, DualG(b, bg)) = DualG(a - b, -bg)
    static member (*) (a, DualG(b, bg)) = DualG(b * a, bg * a)
    static member (/) (a, DualG(b, bg)) = DualG(a / b, -a * bg / (b * b))
    static member Pow (a, DualG(b, bg)) = let apowb = a ** b in DualG(apowb, apowb * (log a) * bg)
    // DualG - int binary operations
    static member (+) (a:DualG, b:int) = a + float b
    static member (-) (a:DualG, b:int) = a - float b
    static member (*) (a:DualG, b:int) = a * float b
    static member (/) (a:DualG, b:int) = a / float b
    static member Pow (a:DualG, b:int) = DualG.Pow(a, float b)
    // int - DualG binary operations
    static member (+) (a:int, b:DualG) = (float a) + b
    static member (-) (a:int, b:DualG) = (float a) - b
    static member (*) (a:int, b:DualG) = (float a) * b
    static member (/) (a:int, b:DualG) = (float a) / b
    static member Pow (a:int, b:DualG) = DualG.Pow(float a, b)
    // DualG unary operations
    static member Log (DualG(a, ag)) = DualG(log a, ag / a)
    static member Exp (DualG(a, ag)) = let expa = exp a in DualG(expa, ag * expa)
    static member Sin (DualG(a, ag)) = DualG(sin a, ag * cos a)
    static member Cos (DualG(a, ag)) = DualG(cos a, -ag * sin a)
    static member Tan (DualG(a, ag)) = let cosa = cos a in DualG(tan a, ag / (cosa * cosa))
    static member (~-) (DualG(a, ag)) = DualG(-a, -ag)
    static member Sqrt (DualG(a, ag)) = let sqrta = sqrt a in DualG(sqrta, ag / (2. * sqrta))
    static member Sinh (DualG(a, ag)) = DualG(sinh a, ag * cosh a)
    static member Cosh (DualG(a, ag)) = DualG(cosh a, ag * sinh a)
    static member Tanh (DualG(a, ag)) = let cosha = cosh a in DualG(tanh a, ag / (cosha * cosha))
    static member Asin (DualG(a, ag)) = DualG(asin a, ag / sqrt (1. - a * a))
    static member Acos (DualG(a, ag)) = DualG(acos a, -ag / sqrt (1. - a * a))
    static member agan (DualG(a, ag)) = DualG(atan a, ag / (1. + a * a))

/// DualG operations module (automatically opened)
[<AutoOpen>]
module DualGOps =
    /// Make DualG, with primal value `p`, gradient dimension `m`, and all gradient components 0
    let inline dualG p m = DualG(p, Vector.Create(m, 0.))
    /// Make DualG, with primal value `p` and gradient array `g`
    let inline dualGSet (p, g) = DualG(p, Vector.Create(g))
    /// Make active DualG (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the component with index `i` having value 1, and the rest of the components 0
    let inline dualGAct p m i = DualG(p, Vector.Create(m, i, 1.))
    /// Make an array of active DualG, with primal values given in array `x`. For a DualG with index _i_, the gradient is the unit vector with 1 in the _i_th place.
    let inline dualGActArray (x:float[]) = Array.init x.Length (fun i -> dualGAct x.[i] x.Length i)
    /// Get the primal value of a DualG
    let inline primal (DualG(p, _)) = p
    /// Get the gradient array of a DualG
    let inline gradient (DualG(_, g)) = array g
    /// Get the first gradient component of a DualG
    let inline tangent (DualG(_, g)) = g.FirstItem
    /// Get the primal value and the first gradient component of a DualG, as a tuple
    let inline tuple (DualG(p, g)) = (p, g.FirstItem)
    /// Get the primal value and the gradient array of a DualG, as a tuple
    let inline tupleG (DualG(p, g)) = (p, array g)


/// ForwardG differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardGOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        dualGAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        dualGAct x 1 0 |> f |> tangent

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        dualGActArray x |> f |> tupleG

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        let a = dualGActArray x |> f
        (Array.map primal a, array2d (Matrix.Create(a.Length, fun i -> gradient a.[i])))

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let (v, j) = jacobian' f x in (v, transpose j)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x = ForwardGOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x = ForwardGOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x = ForwardGOps.grad' f (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x = ForwardGOps.grad f (array x) |> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x = ForwardGOps.jacobianT' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x = ForwardGOps.jacobianT f (array x) |> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x = ForwardGOps.jacobian' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x = ForwardGOps.jacobian f (array x) |> matrix
