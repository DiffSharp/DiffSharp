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

#light

/// Forward AD module
module DiffSharp.AD.Forward

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// Dual numeric type, keeping primal and tangent values
[<CustomEquality; CustomComparison>]
type Dual =
    // Primal, tangent
    | Dual of float * float
    override d.ToString() = let (Dual(p, t)) = d in sprintf "Dual(%f, %f)" p t
    static member op_Explicit(p) = Dual(p, 0.)
    static member op_Explicit(Dual(p, _)) = p
    static member DivideByInt(Dual(p, t), i:int) = Dual(p / float i, t / float i)
    static member Zero = Dual(0., 0.)
    static member One = Dual(1., 0.)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? Dual as d2 -> let Dual(a, _), Dual(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this Dual with another type of object."
    override d.Equals(other) = 
        match other with
        | :? Dual as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (Dual(a, b)) = d in hash [|a; b|]
    // Dual - Dual binary operations
    static member (+) (Dual(a, at), Dual(b, bt)) = Dual(a + b, at + bt)
    static member (-) (Dual(a, at), Dual(b, bt)) = Dual(a - b, at - bt)
    static member (*) (Dual(a, at), Dual(b, bt)) = Dual(a * b, at * b + a * bt)
    static member (/) (Dual(a, at), Dual(b, bt)) = Dual(a / b, (at * b - a * bt) / (b * b))
    static member Pow (Dual(a, at), Dual(b, bt)) = let apb = a ** b in Dual(apb, apb * ((b * at / a) + ((log a) * bt)))
    static member Atan2 (Dual(a, at), Dual(b, bt)) = Dual(atan2 a b, (at * b - a * bt) / (a * a + b * b))
    // Dual - float binary operations
    static member (+) (Dual(a, at), b) = Dual(a + b, at)
    static member (-) (Dual(a, at), b) = Dual(a - b, at)
    static member (*) (Dual(a, at), b) = Dual(a * b, at * b)
    static member (/) (Dual(a, at), b) = Dual(a / b, at / b)
    static member Pow (Dual(a, at), b) = Dual(a ** b, b * (a ** (b - 1.)) * at)
    static member Atan2 (Dual(a, at), b) = Dual(atan2 a b, (b * at) / (b * b + a * a))
    // float - Dual binary operations
    static member (+) (a, Dual(b, bt)) = Dual(b + a, bt)
    static member (-) (a, Dual(b, bt)) = Dual(a - b, -bt)
    static member (*) (a, Dual(b, bt)) = Dual(b * a, bt * a)
    static member (/) (a, Dual(b, bt)) = Dual(a / b, -a * bt / (b * b))
    static member Pow (a, Dual(b, bt)) = let apb = a ** b in Dual(apb, apb * (log a) * bt)
    static member Atan2 (a, Dual(b, bt)) = Dual(atan2 a b, -(a * bt) / (a * a + b * b))
    // Dual - int binary operations
    static member (+) (a:Dual, b:int) = a + float b
    static member (-) (a:Dual, b:int) = a - float b
    static member (*) (a:Dual, b:int) = a * float b
    static member (/) (a:Dual, b:int) = a / float b
    static member Pow (a:Dual, b:int) = Dual.Pow(a, float b)
    static member Atan2 (a:Dual, b:int) = Dual.Atan2(a, float b)
    // int - Dual binary operations
    static member (+) (a:int, b:Dual) = (float a) + b
    static member (-) (a:int, b:Dual) = (float a) - b
    static member (*) (a:int, b:Dual) = (float a) * b
    static member (/) (a:int, b:Dual) = (float a) / b
    static member Pow (a:int, b:Dual) = Dual.Pow(float a, b)
    static member Atan2 (a:int, b:Dual) = Dual.Atan2(float a, b)
    // Dual unary operations
    static member Log (Dual(a, at)) = Dual(log a, at / a)
    static member Log10 (Dual(a, at)) = Dual(log10 a, at / (a * log10val))
    static member Exp (Dual(a, at)) = let expa = exp a in Dual(expa, at * expa)
    static member Sin (Dual(a, at)) = Dual(sin a, at * cos a)
    static member Cos (Dual(a, at)) = Dual(cos a, -at * sin a)
    static member Tan (Dual(a, at)) = let cosa = cos a in Dual(tan a, at / (cosa * cosa))
    static member (~-) (Dual(a, at)) = Dual(-a, -at)
    static member Sqrt (Dual(a, at)) = let sqrta = sqrt a in Dual(sqrta, at / (2. * sqrta))
    static member Sinh (Dual(a, at)) = Dual(sinh a, at * cosh a)
    static member Cosh (Dual(a, at)) = Dual(cosh a, at * sinh a)
    static member Tanh (Dual(a, at)) = let cosha = cosh a in Dual(tanh a, at / (cosha * cosha))
    static member Asin (Dual(a, at)) = Dual(asin a, at / sqrt (1. - a * a))
    static member Acos (Dual(a, at)) = Dual(acos a, -at / sqrt (1. - a * a))
    static member Atan (Dual(a, at)) = Dual(atan a, at / (1. + a * a))
    static member Abs (Dual(a, at)) = 
        if a = 0. then invalidArg "" "The derivative of abs is not defined at 0."
        Dual(abs a, at * float (sign a))
    static member Floor (Dual(a, _)) =
        if isInteger a then invalidArg "" "The derivative of floor is not defined for integer values."
        Dual(floor a, 0.)
    static member Ceiling (Dual(a, _)) =
        if isInteger a then invalidArg "" "The derivative of ceil is not defined for integer values."
        Dual(ceil a, 0.)
    static member Round (Dual(a, _)) =
        if isHalfway a then invalidArg "" "The derivative of round is not defined for values halfway between integers."
        Dual(round a, 0.)

/// Dual operations module (automatically opened)
[<AutoOpen>]
module DualOps =
    /// Make Dual, with primal value `p` and tangent 0
    let inline dual p = Dual(p, 0.)
    /// Make Dual, with primal value `p` and tangent value `t`
    let inline dualSet (p, t) = Dual(p, t)
    /// Make active Dual (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualAct p = Dual(p, 1.)
    /// Make an array of arrays of Dual, with primal values given in array `x`. The tangent values along the diagonal are 1 and the rest are 0.
    let inline dualActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dualAct x.[j] else dual x.[j])))
    /// Get the primal value of a Dual
    let inline primal (Dual(p, _)) = p
    /// Get the tangent value of a Dual
    let inline tangent (Dual(_, t)) = t
    /// Get the primal and tangent values of a Dual, as a tuple
    let inline tuple (Dual(p, t)) = (p, t)


/// Forward differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        dualAct x |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        dualAct x |> f |> tangent
       
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        Array.zip x v |> Array.map dualSet |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        let a = Array.map f (dualActArrayArray x)
        (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let a = Array.map f (dualActArrayArray x)
        (Array.map primal a.[0], Array2D.map tangent (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v = 
        Array.zip x v |> Array.map dualSet |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v = 
        jacobianv' f x v |> snd
            
/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:Dual->Dual) (x:float) = ForwardOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:Dual->Dual) (x:float) = ForwardOps.diff f x
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<Dual>->Dual) (x:Vector<float>) (v:Vector<float>) = ForwardOps.gradv' (vector >> f) (array x) (array v)
    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<Dual>->Dual) (x:Vector<float>) (v:Vector<float>) = ForwardOps.gradv (vector >> f) (array x) (array v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<Dual>->Dual) (x:Vector<float>) = ForwardOps.grad' (vector >> f) (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<Dual>->Dual) (x:Vector<float>) = ForwardOps.grad (vector >> f) (array x) |> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) = ForwardOps.jacobianT' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) = ForwardOps.jacobianT (vector >> f >> array) (array x) |> Matrix.ofArray2d
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) = ForwardOps.jacobian' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) = ForwardOps.jacobian (vector >> f >> array) (array x) |> Matrix.ofArray2d
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) (v:Vector<float>) = ForwardOps.jacobianv' (vector >> f >> array) (array x) (array v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<Dual>->Vector<Dual>) (x:Vector<float>) (v:Vector<float>) = ForwardOps.jacobianv (vector >> f >> array) (array x) (array v) |> vector

/// Numeric literal for a Dual with tangent 0
module NumericLiteralQ = // (Allowed literals : Q, R, Z, I, N, G)
    let FromZero () = dual 0.
    let FromOne () = dual 1.
    let FromInt32 p = dual (float p)


/// Numeric literal for a Dual with tangent 1 (i.e. the variable of differentiation)
module NumericLiteralR =    
    let FromZero () = dualAct 0.
    let FromOne () = dualAct 1.
    let FromInt32 p = dualAct (float p)
