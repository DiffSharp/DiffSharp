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
// Reverse mode AD
// Builds up a trace in forward evaluation, uses this in the reverse sweep for updating adjoint values
// 

#light

/// Reverse AD module
module DiffSharp.AD.Reverse

open System.Collections.Generic
open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// Global trace for recording operations
type Trace() =
    static let mutable stack = new Stack<Op>()
    static member Stack
        with get() = stack
        and set s = stack <- s
    static member Push = Trace.Stack.Push
    static member Clear = Trace.Stack.Clear
    static member Copy() = new Stack<Op>(Trace.Stack)
    static member Set(t) = Trace.Stack <- t
    static member SetClean(t) = Trace.Stack <- Trace.CleanCopy t
    static member CleanCopy(t) =
        let ret = new Stack<Op>()
        for op in t do
            match op with
            | Add(x, y, z) | Sub(x, y, z) | Mul(x, y, z) | Div(x, y, z) | Pow (x, y, z) -> x.A <- 0.; y.A <- 0.; z.A <- 0.
            | Neg(x, z) | Log(x, z) | Exp (x, z) | Sin(x, z) | Cos(x, z) | Tan(x, z) | Sqrt(x, z) | Sinh(x, z) | Cosh(x, z) | Tanh(x, z) | Asin(x, z) | Acos(x, z) | Atan(x, z) -> x.A <- 0.; z.A <- 0.;
            ret.Push(op)
        ret
    static member ReverseSweep() =
        while Trace.Stack.Count > 0 do
            match Trace.Stack.Pop() with
            | Add(x, y, z) -> x.AddAdj(z.A); y.AddAdj(z.A)
            | Sub(x, y, z) -> x.AddAdj(z.A); y.AddAdj(-z.A)
            | Mul(x, y, z) -> x.AddAdj(z.A * y.P); y.AddAdj(z.A * x.P)
            | Div(x, y, z) -> x.AddAdj(z.A * (1. / y.P)); y.AddAdj(z.A * (-x.P / (y.P * y.P)))
            | Pow(x, y, z) -> x.AddAdj(z.A * (x.P ** (y.P - 1.)) * y.P); y.AddAdj(z.A * (x.P ** y.P) * log x.P)
            | Log(x, z) -> x.AddAdj(z.A / x.P)
            | Exp(x, z) -> x.AddAdj(z.A * z.P)
            | Sin(x, z) -> x.AddAdj(z.A * cos x.P)
            | Cos(x, z) -> x.AddAdj(z.A * (-sin x.P))
            | Tan(x, z) -> let secx = 1. / cos x.P in x.AddAdj(z.A * (secx * secx))
            | Neg(x, z) -> x.AddAdj(-z.A)
            | Sqrt(x, z) -> x.AddAdj(z.A / (2. * z.P))
            | Sinh(x, z) -> x.AddAdj(z.A * cosh x.P)
            | Cosh(x, z) -> x.AddAdj(z.A * sinh x.P)
            | Tanh(x, z) -> let sechx = 1. / cosh x.P in x.AddAdj(z.A * (sechx * sechx))
            | Asin(x, z) -> x.AddAdj(z.A / sqrt (1. - x.P * x.P))
            | Acos(x, z) -> x.AddAdj(-z.A / sqrt (1. - x.P * x.P))
            | Atan(x, z) -> x.AddAdj(z.A / (1. + x.P * x.P))

/// Discriminated union of operations for recording the trace
and Op =
    | Add of Adj * Adj * Adj
    | Sub of Adj * Adj * Adj
    | Mul of Adj * Adj * Adj
    | Div of Adj * Adj * Adj
    | Pow of Adj * Adj * Adj
    | Log of Adj * Adj
    | Exp of Adj * Adj
    | Sin of Adj * Adj
    | Cos of Adj * Adj
    | Tan of Adj * Adj
    | Neg of Adj * Adj
    | Sqrt of Adj * Adj
    | Sinh of Adj * Adj
    | Cosh of Adj * Adj
    | Tanh of Adj * Adj
    | Asin of Adj * Adj
    | Acos of Adj * Adj
    | Atan of Adj * Adj

/// Adj numeric type, keeping primal and adjoint values
and Adj =
    val P:float // Primal
    val mutable A:float // Adjoint
    new(p) = {P = p; A = 0.}
    new(p, a) = {P = p; A = a}
    override this.ToString() = sprintf "Adj(%f, %f)" this.P this.A
    static member op_Explicit(x) = Adj(x)
    static member op_Explicit(x:Adj) = x.P
    static member DivideByInt(x:Adj, i:int) = Adj(x.P / float i, x.A / float i)
    static member Zero = Adj(0., 0.)
    static member One = Adj(1., 0.)
    member this.AddAdj(a) = this.A <- this.A + a
    // Adj - Adj binary operations
    static member (+) (x:Adj, y:Adj) = let z = Adj(x.P + y.P) in Trace.Push(Add(x, y, z)); z
    static member (-) (x:Adj, y:Adj) = let z = Adj(x.P - y.P) in Trace.Push(Sub(x, y, z)); z
    static member (*) (x:Adj, y:Adj) = let z = Adj(x.P * y.P) in Trace.Push(Mul(x, y, z)); z
    static member (/) (x:Adj, y:Adj) = let z = Adj(x.P / y.P) in Trace.Push(Div(x, y, z)); z
    static member Pow (x:Adj, y:Adj) = let z = Adj(x.P ** y.P) in Trace.Push(Pow(x, y, z)); z
    // Adj - float binary operations
    static member (+) (x:Adj, y:float) = x + Adj(y)
    static member (-) (x:Adj, y:float) = x - Adj(y)
    static member (*) (x:Adj, y:float) = x * Adj(y)
    static member (/) (x:Adj, y:float) = x / Adj(y)
    static member Pow (x:Adj, y:float) = x ** Adj(y)
    // float - Adj binary operations
    static member (+) (x:float, y:Adj) = Adj(x) + y
    static member (-) (x:float, y:Adj) = Adj(x) - y
    static member (*) (x:float, y:Adj) = Adj(x) * y
    static member (/) (x:float, y:Adj) = Adj(x) / y
    static member Pow (x:float, y:Adj) = Adj(x) ** y
    // Adj - int binary operations
    static member (+) (x:Adj, y:int) = x + Adj(float y)
    static member (-) (x:Adj, y:int) = x - Adj(float y)
    static member (*) (x:Adj, y:int) = x * Adj(float y)
    static member (/) (x:Adj, y:int) = x / Adj(float y)
    static member Pow (x:Adj, y:int) = x ** Adj(float y)
    // int - Adj binary operations
    static member (+) (x:int, y:Adj) = Adj(float x) + y
    static member (-) (x:int, y:Adj) = Adj(float x) - y
    static member (*) (x:int, y:Adj) = Adj(float x) * y
    static member (/) (x:int, y:Adj) = Adj(float x) / y
    static member Pow (x:int, y:Adj) = Adj(float x) ** y
    // Adj unary operations
    static member Log (x:Adj) = let z = Adj(log x.P) in Trace.Push(Log(x, z)); z
    static member Exp (x:Adj) = let z = Adj(exp x.P) in Trace.Push(Exp(x, z)); z
    static member Sin (x:Adj) = let z = Adj(sin x.P) in Trace.Push(Sin(x, z)); z
    static member Cos (x:Adj) = let z = Adj(cos x.P) in Trace.Push(Cos(x, z)); z
    static member Tan (x:Adj) = let z = Adj(tan x.P) in Trace.Push(Tan(x, z)); z
    static member (~-) (x:Adj) = let z = Adj(-x.P) in Trace.Push(Neg(x, z)); z
    static member Sqrt (x:Adj) = let z = Adj(sqrt x.P) in Trace.Push(Sqrt(x, z)); z
    static member Sinh (x:Adj) = let z = Adj(sinh x.P) in Trace.Push(Sinh(x, z)); z
    static member Cosh (x:Adj) = let z = Adj(cosh x.P) in Trace.Push(Cosh(x, z)); z
    static member Tanh (x:Adj) = let z = Adj(tanh x.P) in Trace.Push(Tanh(x, z)); z
    static member Asin (x:Adj) = let z = Adj(asin x.P) in Trace.Push(Asin(x, z)); z
    static member Acos (x:Adj) = let z = Adj(acos x.P) in Trace.Push(Acos(x, z)); z
    static member Atan (x:Adj) = let z = Adj(atan x.P) in Trace.Push(Atan(x, z)); z

/// Adj operations module (automatically opened)
[<AutoOpen>]
module AdjOps =
    /// Make Adj, with primal value `p` and adjoint 0
    let adj p = Adj(p)
    /// Get the primal value of an Adj
    let primal (x:Adj) = x.P
    /// Get the adjoint value of an Adj
    let adjoint (x:Adj) = x.A

/// Reverse differentiation operations module (automatically opened)
[<AutoOpen>]
module ReverseOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        Trace.Clear()
        let xa = adj x
        let z:Adj = f xa
        z.A <- 1.
        Trace.ReverseSweep()
        (primal z, adjoint xa)

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        Trace.Clear()
        let xa = Array.map adj x
        let z:Adj = f xa
        z.A <- 1.
        Trace.ReverseSweep()
        (primal z, Array.map adjoint xa)

    /// Original value of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`, using one forward sweep and several reverse sweeps
    let inline jacobian' f x =
        Trace.Clear()
        let xa = Array.map adj x
        let z:Adj[] = f xa
        let forwardTrace = Trace.Copy()
        let jac = Array.init z.Length (fun i ->
                                        if i > 0 then Trace.SetClean(forwardTrace)
                                        z.[i].A <- 1.
                                        Trace.ReverseSweep()
                                        Array.map adjoint xa)
        (Array.map primal z, array2D jac)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let (v, j) = jacobian' f x in (v, transpose j)
    
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline gradhessian' f x =
        let xv = Vector(x)
        let (v, g) = grad' f x
        let h = Matrix.Create(x.Length, g)
        let hh = Matrix.Create(x.Length, fun i -> grad f (array (xv + Vector.Create(x.Length, i, eps))))
        (v, g, array2d ((hh - h) / eps))

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd
    
    /// Original valuea and Hessian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline hessian' f x =
        gradhessian' f x |> fsttrd

    /// Hessian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline hessian f x =
        gradhessian' f x |> trd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline laplacian' f x =
        let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`, using finite differences over reverse mode gradient
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv' f x (v:float[]) =
        Trace.Clear()
        let xa = Array.map adj x
        let z:Adj[] = f xa
        for i = 0 to z.Length - 1 do z.[i].A <- v.[i]
        Trace.ReverseSweep()
        (Array.map primal z, Array.map adjoint xa)

    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv f x v =
        jacobianTv' f x v |> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:Adj->Adj) (x:float) = ReverseOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:Adj->Adj) (x:float) = ReverseOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.grad' (vector >> f) (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.grad (vector >> f) (array x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.laplacian' (vector >> f) (array x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.laplacian (vector >> f) (array x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<Adj>->Vector<Adj>) (x:Vector<float>) = ReverseOps.jacobianT' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<Adj>->Vector<Adj>) (x:Vector<float>) = ReverseOps.jacobianT (vector >> f >> array) (array x) |> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<Adj>->Vector<Adj>) (x:Vector<float>) = ReverseOps.jacobian' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<Adj>->Vector<Adj>) (x:Vector<float>) = ReverseOps.jacobian (vector >> f >> array) (array x) |> matrix
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.hessian' (vector >> f) (array x) |> fun (a, b) -> (a, matrix b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.hessian (vector >> f) (array x) |> matrix
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.gradhessian' (vector >> f) (array x) |> fun (a, b, c) -> (a, vector b, matrix c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<Adj>->Adj) (x:Vector<float>) = ReverseOps.gradhessian (vector >> f) (array x) |> fun (a, b) -> (vector a, matrix b)