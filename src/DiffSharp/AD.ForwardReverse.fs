//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (C) 2014, 2015, National University of Ireland Maynooth.
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

/// Reverse-on-forward mode AD module
module DiffSharp.AD.ForwardReverse

open DiffSharp.AD.Reverse
open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualAdj numeric type, keeping the primals and adjoints of primal and tangent values
[<CustomEquality; CustomComparison>]
type DualAdj =
    // Primal, tangent
    | DualAdj of Adj * Adj
    override d.ToString() = let (DualAdj(p, t)) = d in sprintf "DualAdj(%A, %A, %A, %A)" p.P p.A t.P t.A
    /// The adjoint of the primal of `d`
    member d.A
        with set(a) = let (DualAdj(p, _)) = d in p.A <- a
    /// The adjoint of the tangent of `d`
    member d.TA
        with set(a) = let (DualAdj(_, t)) = d in t.A <- a
    static member op_Explicit(p) = DualAdj(p, adj 0.)
    static member op_Explicit(DualAdj(p, _)) = p.P
    static member DivideByInt(DualAdj(p, t), i:int) = DualAdj(p / adj i, t / adj i)
    static member Zero = DualAdj(adj 0., adj 0.)
    static member One = DualAdj(adj 1, adj 0.)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualAdj as d2 -> let DualAdj(a, _), DualAdj(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualAdj with another type of object."
    override d.Equals(other) =
        match other with
        | :? DualAdj as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualAdj(a, b)) = d in hash [|a; b|]
    // DualAdj - DualAdj binary operations
    static member (+) (DualAdj(a, at), DualAdj(b, bt)) = DualAdj(a + b, at + bt)
    static member (-) (DualAdj(a, at), DualAdj(b, bt)) = DualAdj(a - b, at - bt)
    static member (*) (DualAdj(a, at), DualAdj(b, bt)) = DualAdj(a * b, at * b + a * bt)
    static member (/) (DualAdj(a, at), DualAdj(b, bt)) = DualAdj(a / b, (at * b - a * bt) / (b * b))
    static member Pow (DualAdj(a, at), DualAdj(b, bt)) = let apb = a ** b in DualAdj(apb, apb * ((b * at / a) + ((log a) * bt)))
    static member Atan2 (DualAdj(a, at), DualAdj(b, bt)) = DualAdj(atan2 a b, (at * b - a * bt) / (a * a + b * b))
    // DualAdj - float binary operations
    static member (+) (DualAdj(a, at), b:float) = DualAdj(a + b, at)
    static member (-) (DualAdj(a, at), b:float) = DualAdj(a - b, at)
    static member (*) (DualAdj(a, at), b:float) = DualAdj(a * b, at * b)
    static member (/) (DualAdj(a, at), b:float) = DualAdj(a / b, at / b)
    static member Pow (DualAdj(a, at), b:float) = DualAdj(a ** b, b * (a ** (b - 1.)) * at)
    static member Atan2 (DualAdj(a, at), b:float) = DualAdj(atan2 a (adj b), (b * at) / (b * b + a * a))
    // float - DualAdj binary operations
    static member (+) (a:float, DualAdj(b, bt)) = DualAdj(b + a, bt)
    static member (-) (a:float, DualAdj(b, bt)) = DualAdj(a - b, -bt)
    static member (*) (a:float, DualAdj(b, bt)) = DualAdj(b * a, bt * a)
    static member (/) (a:float, DualAdj(b, bt)) = DualAdj(a / b, -a * bt / (b * b))
    static member Pow (a:float, DualAdj(b, bt)) = let apb = (adj a) ** b in DualAdj(apb, apb * (log a) * bt)
    static member Atan2 (a:float, DualAdj(b, bt)) = DualAdj(atan2 (adj a) b, -(a * bt) / (a * a + b * b))
    // DualAdj - int binary operations
    static member (+) (a:DualAdj, b:int) = a + float b
    static member (-) (a:DualAdj, b:int) = a - float b
    static member (*) (a:DualAdj, b:int) = a * float b
    static member (/) (a:DualAdj, b:int) = a / float b
    static member Pow (a:DualAdj, b:int) = DualAdj.Pow(a, float b)
    static member Atan2 (a:DualAdj, b:int) = DualAdj.Atan2(a, float b)
    // int - DualAdj binary operations
    static member (+) (a:int, b:DualAdj) = (float a) + b
    static member (-) (a:int, b:DualAdj) = (float a) - b
    static member (*) (a:int, b:DualAdj) = (float a) * b
    static member (/) (a:int, b:DualAdj) = (float a) / b
    static member Pow (a:int, b:DualAdj) = DualAdj.Pow(float a, b)
    static member Atan2 (a:int, b:DualAdj) = DualAdj.Atan2(float a, b)
    // DualAdj unary operations
    static member Log (DualAdj(a, at)) = 
        if a.P <= 0. then invalidArgLog()
        DualAdj(log a, at / a)
    static member Log10 (DualAdj(a, at)) = 
        if a.P <= 0. then invalidArgLog10()
        DualAdj(log10 a, at / (a * log10val))
    static member Exp (DualAdj(a, at)) = let expa = exp a in DualAdj(expa, at * expa)
    static member Sin (DualAdj(a, at)) = DualAdj(sin a, at * cos a)
    static member Cos (DualAdj(a, at)) = DualAdj(cos a, -at * sin a)
    static member Tan (DualAdj(a, at)) = 
        if cos a.P = 0. then invalidArgTan()
        let cosa = cos a in DualAdj(tan a, at / (cosa * cosa))
    static member (~-) (DualAdj(a, at)) = DualAdj(-a, -at)
    static member Sqrt (DualAdj(a, at)) = 
        if a.P <= 0. then invalidArgSqrt()
        let sqrta = sqrt a in DualAdj(sqrta, at / (2. * sqrta))
    static member Sinh (DualAdj(a, at)) = DualAdj(sinh a, at * cosh a)
    static member Cosh (DualAdj(a, at)) = DualAdj(cosh a, at * sinh a)
    static member Tanh (DualAdj(a, at)) = let cosha = cosh a in DualAdj(tanh a, at / (cosha * cosha))
    static member Asin (DualAdj(a, at)) = 
        if (abs a.P) >= 1. then invalidArgAsin()
        DualAdj(asin a, at / sqrt (1. - a * a))
    static member Acos (DualAdj(a, at)) = 
        if (abs a.P) >= 1. then invalidArgAcos()
        DualAdj(acos a, -at / sqrt (1. - a * a))
    static member Atan (DualAdj(a, at)) = DualAdj(atan a, at / (1. + a * a))
    static member Abs (DualAdj(a, at)) = 
        if a.P = 0. then invalidArgAbs()
        DualAdj(abs a, at * (sign a.P))
    static member Floor (DualAdj(a, _)) =
        if isInteger a.P then invalidArgFloor()
        DualAdj(floor a, adj 0.)
    static member Ceiling (DualAdj(a, _)) =
        if isInteger a.P then invalidArgCeil()
        DualAdj(ceil a, adj 0.)
    static member Round (DualAdj(a, _)) =
        if isHalfway a.P then invalidArgRound()
        DualAdj(round a, adj 0.)

/// DualAdj operations module (automatically opened)
[<AutoOpen>]
module DualAdjOps =
    /// Make DualAdj, with primal value `p` and tangent 0
    let inline dualAdj p = DualAdj(adj p, adj 0.)
    /// Make DualAdj, with primal value `p` and tangent value `t`
    let inline dualAdjSet (p, t) = DualAdj(adj p, adj t)
    /// Make active DualAdj (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualAdjAct p = DualAdj(adj p, adj 1.)
    /// Get the primal value of a DualAdj
    let inline primal (DualAdj(p, _)) = p.P
    /// Get the adjoint of the primal value of a DualAdj
    let inline adjoint (DualAdj(p, _)) = p.A
    /// Get the tangent value of a DualAdj
    let inline tangent (DualAdj(_, t)) = t.P
    /// Get the adjoint of the tangent value of a DualAdj
    let inline tangentAdjoint (DualAdj(_, t)) = t.A
    /// Get the primal and tangent values of a DualAdj, as a tuple
    let inline tuple (DualAdj(p, t)) = (p.P, t.P)


/// ForwardReverse differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardReverseOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`. Computed using forward mode AD.
    let inline diff' f (x:float) =
        dualAdjAct x |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`. Computed using forward mode AD.
    let inline diff f (x:float) =
        dualAdjAct x |> f |> tangent

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2'' f (x:float) =
        Trace.Clear()
        let xa = dualAdjAct x
        let z:DualAdj = f xa
        z.TA <- 1.
        Trace.ReverseSweep()
        (primal z, tangent z, adjoint xa)

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2 f x =
        diff2'' f x |> trd

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2' f x =
        diff2'' f x |> fsttrd

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline gradv' f (x:float[]) (v:float[]) =
        Array.zip x v |> Array.map dualAdjSet |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Computed using reverse mode AD.
    let inline grad' f (x:float[]) =
        Trace.Clear()
        let xa = Array.map dualAdj x
        let z:DualAdj = f xa
        z.A <- 1.
        Trace.ReverseSweep()
        (primal z, Array.map adjoint xa)

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Computed using reverse mode AD.
    let inline grad f x =
        grad' f x |> snd

    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv' f (x:float[]) (v:float[]) =
        Trace.Clear()
        let xa = Array.map dualAdjSet (Array.zip x v)
        let z:DualAdj = f xa
        z.TA <- 1.
        Trace.ReverseSweep()
        (primal z, tangent z, Array.map adjoint xa)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv f x v =
        gradhessianv' f x v |> sndtrd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline gradhessian' f (x:float[]) =
        let a = Array.init x.Length (fun i -> gradhessianv' f x (standardBasis x.Length i))
        (fst3 a.[0], Array.map snd3 a, array2D (Array.map trd a))

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv' f x v =
        gradhessianv' f x v |> fsttrd

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv f x v =
        hessianv' f x v |> snd

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessian' f (x:float[]) =
        let a = Array.init x.Length (fun i -> hessianv' f x (standardBasis x.Length i))
        (fst a.[0], array2D (Array.map snd a))

    /// Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessian f x =
        hessian' f x |> snd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline laplacian' f x =
        let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline jacobianv' f (x:float[]) (v:float[]) =
        Array.zip x v |> Array.map dualAdjSet |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline jacobianv f x v =
        jacobianv' f x v |> snd

    /// Original value, Jacobian-vector product, and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned 3-tuple, the first is the original value of function `f` at point `x`, the second is the Jacobian-vector product of `f` at point `x` along vector `v1` (computed using forward mode AD), and the third is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianvTv'' f (x:float[]) (v1:float[]) =
        Trace.Clear()
        let xa = Array.map dualAdjSet (Array.zip x v1)
        let z:DualAdj[] = f xa
        let forwardTrace = Trace.Copy()
        let r1 = Array.map primal z
        let r2 = Array.map tangent z
        let r3 =
            fun v2 ->
                Trace.SetClean(forwardTrace)
                Array.iter2 (fun (a:DualAdj) b -> a.A <- b) z v2
                Trace.ReverseSweep()
                Array.map adjoint xa
        (r1, r2, r3)

    /// Original value, Jacobian-vector product, and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Jacobian-vector product is computed using forward mode AD, along vector `v1`. Transposed Jacobian-vector product is computed using reverse mode AD, along vector `v2`.
    let inline jacobianvTv' f x v1 v2 =
        let r1, r2, r3 = jacobianvTv'' f x v1
        (r1, r2, r3 v2)

    /// Jacobian-vector product and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Jacobian-vector product is computed using forward mode AD, along vector `v1`. Transposed Jacobian-vector product is computed using reverse mode AD, along vector `v2`.
    let inline jacobianvTv f x v1 v2 =
        jacobianvTv' f x v1 v2 |> sndtrd

    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Computed using reverse mode AD. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' f (x:float[]) =
        let r1, _, r3 = jacobianvTv'' f x (Array.zeroCreate x.Length)
        (r1, r3)

    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using reverse mode AD.
    let inline jacobianTv' f x (v:float[]) =
        let r1, r2 = jacobianTv'' f x
        (r1, r2 v)

    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using reverse mode AD.
    let inline jacobianTv f x v =
        jacobianTv' f x v |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobianT' (f:DualAdj[]->DualAdj[]) (x:float[]) =
        let o = x |> Array.map dualAdj |> f |> Array.map primal
        if x.Length < o.Length then // f:R^n -> R^m, if n < m use forward mode AD
            let a = Array.init x.Length (fun i -> jacobianv f x (standardBasis x.Length i))
            (o, array2D a)
        else // f:R^n -> R^m, if n > m, use reverse mode AD
            let a = Array.init o.Length (fun i -> jacobianTv f x (standardBasis o.Length i))
            (o, transpose (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobian' f x =
        let o, j = jacobianT' f x
        (o, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobian f x =
        jacobian' f x |> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`. Computed using forward mode AD.
    let inline diff' (f:DualAdj->DualAdj) x = ForwardReverseOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`. Computed using forward mode AD.
    let inline diff (f:DualAdj->DualAdj) x = ForwardReverseOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2' (f:DualAdj->DualAdj) x = ForwardReverseOps.diff2' f x
   /// Second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2 (f:DualAdj->DualAdj) x = ForwardReverseOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline diff2'' (f:DualAdj->DualAdj) x = ForwardReverseOps.diff2'' f x
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline gradv' (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline gradv (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Computed using reverse mode AD.
    let inline grad' (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`. Computed using reverse mode AD.
    let inline grad (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline gradhessian' (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2d c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline gradhessian (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv' (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.gradhessianv' (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b, c) -> (a, b, vector c)
    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.gradhessianv (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (a, vector b)
    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv' (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.hessianv' (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (a, vector b)
    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv (f:Vector<DualAdj>->DualAdj) x v = ForwardReverseOps.hessianv (vector >> f) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessian' (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2d b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessian (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline laplacian' (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline laplacian (f:Vector<DualAdj>->DualAdj) x = ForwardReverseOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value, Jacobian-vector product, and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned 3-tuple, the first is the original value of function `f` at point `x`, the second is the Jacobian-vector product of `f` at point `x` along vector `v1` (computed using forward mode AD), and the third is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianvTv'' (f:Vector<DualAdj>->Vector<DualAdj>) x v1 = ForwardReverseOps.jacobianvTv'' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v1) |> fun (a, b, c) -> (vector a, vector b, Vector.toArray >> c >> vector)
    /// Original value, Jacobian-vector product, and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Jacobian-vector product is computed using forward mode AD, along vector `v1`. Transposed Jacobian-vector product is computed using reverse mode AD, along vector `v2`.
    let inline jacobianvTv' (f:Vector<DualAdj>->Vector<DualAdj>) x v1 v2 = ForwardReverseOps.jacobianvTv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v1) (Vector.toArray v2) |> fun (a, b, c) -> (vector a, vector b, vector c)
    /// Jacobian-vector product and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Jacobian-vector product is computed using forward mode AD, along vector `v1`. Transposed Jacobian-vector product is computed using reverse mode AD, along vector `v2`.
    let inline jacobianvTv (f:Vector<DualAdj>->Vector<DualAdj>) x v1 v2 = ForwardReverseOps.jacobianvTv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v1) (Vector.toArray v2) |> fun (a, b) -> (vector a, vector b)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobianT' (f:Vector<DualAdj>->Vector<_>) x = ForwardReverseOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobianT (f:Vector<DualAdj>->Vector<_>) x = ForwardReverseOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobian' (f:Vector<DualAdj>->Vector<_>) x = ForwardReverseOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`. For a function `f:R^n -> R^m`, the Jacobian is computed using forward mode AD if n < m, or reverse mode AD if n > m.
    let inline jacobian (f:Vector<DualAdj>->Vector<_>) x = ForwardReverseOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline jacobianv' (f:Vector<DualAdj>->Vector<DualAdj>) x v = ForwardReverseOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using forward mode AD.
    let inline jacobianv (f:Vector<DualAdj>->Vector<DualAdj>) x v = ForwardReverseOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using reverse mode AD.
    let inline jacobianTv (f:Vector<DualAdj>->Vector<DualAdj>) x v = ForwardReverseOps.jacobianTv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Computed using reverse mode AD.
    let inline jacobianTv' (f:Vector<DualAdj>->Vector<DualAdj>) x v = ForwardReverseOps.jacobianTv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Computed using reverse mode AD. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' (f:Vector<DualAdj>->Vector<DualAdj>) x = ForwardReverseOps.jacobianTv'' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Vector.toArray >> b >> vector)