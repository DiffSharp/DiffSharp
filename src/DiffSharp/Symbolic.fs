//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
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
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Symbolic differentiation
//
// - Limited to closed-form algebraic functions, i.e. no control flow
// - Can drill into method bodies of other functions called from the current one, provided these have the [<ReflectedDefinition>] attribute set
// - Can compute higher order derivatives and all combinations of partial derivatives
//

#light

/// Symbolic differentiation
namespace DiffSharp.Symbolic

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open FSharp.Quotations.Evaluator
open DiffSharp.Util
open FsAlg.Generic

/// Symbolic differentiation expression operations module (automatically opened)
[<AutoOpen>]
module ExprOps =

    /// Recursively traverse and differentiate Expr `expr` with respect to Var `v`
    let rec diffExpr (v:Var) expr =
        match expr with
        | Double(_) -> <@@ 0. @@>
        | SpecificCall <@ (+) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float) + %%bt @@>
        | SpecificCall <@ (-) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float) - %%bt @@>
        | SpecificCall <@ (*) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float) * %%b + %%a * %%bt @@>
        | SpecificCall <@ (/) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%at:float) * %%b - %%a * %%bt) / (%%b * %%b)@@>
        | SpecificCall <@ op_Exponentiation @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%a:float) ** %%b) * ((%%b * %%at / %%a) + ((log %%a) * %%bt)) @@>
        | SpecificCall <@ atan2 @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%at:float) * %%b - %%a * %%bt) / (%%a * %%a + %%b * %%b) @@>
        | SpecificCall <@ (~-) @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float) @@>
        | SpecificCall <@ log @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / %%a @@>
        | SpecificCall <@ log10 @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / (%%a * log10val) @@>
        | SpecificCall <@ exp @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * %%expr @@>
        | SpecificCall <@ sin @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * cos %%a @@>
        | SpecificCall <@ cos @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float) * sin %%a @@>
        | SpecificCall <@ tan @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / ((cos %%a) * (cos %%a)) @@>
        | SpecificCall <@ sqrt @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / (2. * %%expr) @@>
        | SpecificCall <@ sinh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * cosh %%a @@>
        | SpecificCall <@ cosh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * sinh %%a @@>
        | SpecificCall <@ tanh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / ((cosh %%a) * (cosh %%a)) @@>
        | SpecificCall <@ asin @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / sqrt (1. - %%a * %%a) @@>
        | SpecificCall <@ acos @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float) / sqrt (1. - %%a * %%a) @@>
        | SpecificCall <@ atan @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / (1. + %%a * %%a) @@>
        | SpecificCall <@ abs @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * float (sign %%a) @@> //The derivative of abs is not defined at 0.
        | SpecificCall <@ floor @> (_, _, [a]) -> <@@ 0. @@> // The derivative of floor is not defined for integer values.
        | SpecificCall <@ ceil @> (_, _, [a]) -> <@@ 0. @@> // The derivative of ceil is not defined for integer values.
        | SpecificCall <@ round @> (_, _, [a]) -> <@@ 0. @@> // The derivative of round is not defined for values halfway between integers.
        | ShapeVar(var) -> if var = v then <@@ 1. @@> else <@@ 0. @@>
        | ShapeLambda(arg, body) -> Expr.Lambda(arg, diffExpr v body)
        | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map (diffExpr v) args)

    /// Simplify Expr `expr`
    let simplify expr =
        let rec simplifyExpr expr =
            match expr with
            | SpecificCall <@ (*) @> (_, _, [a; Double(0.)]) -> <@@ 0. @@>
            | SpecificCall <@ (*) @> (_, _, [Double(0.); b]) -> <@@ 0. @@>
            | SpecificCall <@ (*) @> (_, _, [a; Double(1.)]) -> a
            | SpecificCall <@ (*) @> (_, _, [Double(1.); b]) -> b
            | SpecificCall <@ (+) @> (_, _, [a; Double(0.)]) -> a
            | SpecificCall <@ (+) @> (_, _, [Double(0.); b]) -> b
            | SpecificCall <@ (-) @> (_, _, [a; Double(0.)]) -> a
            | SpecificCall <@ (-) @> (_, _, [Double(0.); b]) -> <@@ -(%%b:float) @@>
            | SpecificCall <@ (/) @> (_, _, [a; Double(1.)]) -> a
            | SpecificCall <@ op_Exponentiation @> (_, _, [Double(1.); _]) -> <@@ 1. @@>
            | ShapeVar(var) -> Expr.Var(var)
            | ShapeLambda(arg, body) -> Expr.Lambda(arg, simplifyExpr body)
            | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map simplifyExpr args)
        let s = Seq.unfold (fun s ->
                        let e = simplifyExpr (fst s)
                        let el = e.ToString().Length
                        if el <> (snd s) then
                            Some(e, (e, el))
                        else
                            None
                        ) (expr, expr.ToString().Length)
        if Seq.isEmpty s then expr else Seq.last s

    /// Completely expand Expr `expr`
    let expand expr =
        let rec expandExpr vars expr = 
            let expanded = 
                match expr with
                | ShapeVar v when Map.containsKey v vars -> vars.[v]
                | ShapeVar v -> Expr.Var v
                | Call(body, MethodWithReflectedDefinition meth, args) ->
                    let this = match body with Some b -> Expr.Application(meth, b) | _ -> meth
                    let res = Expr.Applications(this, [for a in args -> [a]])
                    expandExpr vars res
                | ShapeLambda(v, expr) -> Expr.Lambda(v, expandExpr vars expr)
                | ShapeCombination(o, exprs) -> RebuildShapeCombination(o, List.map (expandExpr vars) exprs)
            match expanded with
            | Application(ShapeLambda(v, body), assign)
            | Let(v, assign, body) -> expandExpr (Map.add v (expandExpr vars assign) vars) body
            | _ -> expanded
        expandExpr Map.empty expr

    /// Get the arguments of a function given in Expr `expr`, as a Var array
    let getExprArgs expr = 
        let rec getLambdaArgs (args:Var[]) = function
            | Lambda(arg, body) -> getLambdaArgs (Array.append args [|arg|]) body
            | _ -> args
        getLambdaArgs Array.empty expr

    /// Compute the `n`-th derivative of an Expr, with respect to Var `v`
    let rec diffExprN v n =
        match n with
        | a when a < 0 -> failwith "Order of derivative cannot be negative."
        | 0 -> fun (x:Expr) -> x
        | 1 -> fun x -> diffExpr v x
        | _ -> fun x -> diffExprN v (n - 1) (diffExpr v x)

    /// Evaluate scalar-to-scalar Expr `expr`, at point `x`
    let evalSS (x:float) expr =
        Expr.Application(expr, Expr.Value(x))
        |> QuotationEvaluator.CompileUntyped
        :?> float

    /// Evaluate vector-to-scalar Expr `expr`, at point `x`
    let evalVS (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float
    
    /// Evaluate vector-to-vector Expr `expr`, at point `x`
    let evalVV (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float[]


/// Symbolic differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// First derivative of a scalar-to-scalar function `f`
    let diff (f:Expr<float->float>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExpr args.[0] fe
        |> simplify
        |> QuotationEvaluator.CompileUntyped
        :?> (float->float)

    /// Original value and first derivative of a scalar-to-scalar function `f`
    let diff' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff f
        fun x -> (fe x, fd x)

    /// `n`-th derivative of a scalar-to-scalar function `f`
    let diffn n (f:Expr<float->float>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExprN args.[0] n fe
        |> simplify
        |> QuotationEvaluator.CompileUntyped
        :?> (float->float)

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`
    let diffn' n f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diffn n f
        fun x -> (fe x, fd x)
    
    /// Second derivative of a scalar-to-scalar function `f`
    let diff2 f =
        diffn 2 f

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let diff2' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff2 f
        fun x -> (fe x, fd x)

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff2'' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff f
        let fd2 = diff2 f
        fun x -> (fe, fd x, fd2 x)

    /// Gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad (f:Expr) =
        let fe = expand f
        let fg = Array.map (fun a -> simplify (diffExpr a fe)) (getExprArgs fe)
        fun x -> Array.map (evalVS x) fg
    
    /// Original value and gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad' f =
        let fg = grad f
        fun x -> (evalVS x f, fg x)

    /// Transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT (f:Expr) =
        let fe = expand f
        let fj = Array.map (fun a -> simplify (diffExpr a fe)) (getExprArgs fe)
        fun x -> Array.map (evalVV x) fj |> array2D

    /// Original value and transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT' f =
        let fj = jacobianT f
        fun x -> (evalVV x f, fj x)

    /// Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian f =
        let fj = jacobianT f
        fun x -> fj x |> transpose

    /// Original value and Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian' f =
        let fj = jacobianT' f
        fun x -> fj x |> fun (r, j) -> (r, transpose j)

    /// Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian (f:Expr) =
        let fe = expand f
        let fd = Array.map (fun a -> simplify (diffExpr a (diffExpr a fe))) (getExprArgs fe)
        fun x -> Array.sumBy (evalVS x) fd

    /// Original value and Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian' f =
        let fd = laplacian f
        fun x -> (evalVS x f, fd x)

    /// Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian (f:Expr) =
        let fe = expand f
        let args = getExprArgs fe
        let fd:Expr[,] = Array2D.create args.Length args.Length (Expr.Value(0.))
        for i = 0 to args.Length - 1 do
            let di = diffExpr args.[i] fe
            for j = i to args.Length - 1 do
                fd.[i, j] <- simplify (diffExpr args.[j] di)
        fun (x:float[]) ->
            let ret:float[,] = Array2D.create x.Length x.Length 0.
            for i = 0 to x.Length - 1 do
                for j = i to x.Length - 1 do
                    ret.[i, j] <- evalVS x fd.[i, j]
            copyUpperToLower ret

    /// Original value and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian' f =
        let fd = hessian f
        fun x -> (evalVS x f, fd x)

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian' f =
        let fh = hessian f
        let fg = grad f
        fun x -> (evalVS x f, fg x, fh x)

    /// Gradient and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian f =
        let fgh = gradhessian' f
        fun x -> fgh x |> sndtrd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        v, trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        let j = jacobianT f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        trace j

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f = DiffOps.diff' f
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f = DiffOps.diff f
    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f = DiffOps.diff2' f
    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f = DiffOps.diff2 f
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff2'' f = DiffOps.diff2'' f
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn' n f = DiffOps.diffn' n f
    /// `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn n f = DiffOps.diffn n f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = Vector.toArray >> DiffOps.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f x = Vector.toArray >> DiffOps.grad f >> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f x = Vector.toArray >> DiffOps.laplacian' f
    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f x = Vector.toArray >> DiffOps.laplacian f
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f x = Vector.toArray >> DiffOps.jacobianT' f >> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f x = Vector.toArray >> DiffOps.jacobianT f >> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f x = Vector.toArray >> DiffOps.jacobian' f >> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f x = Vector.toArray >> DiffOps.jacobian f >> Matrix.ofArray2D
    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f x = Vector.toArray >> DiffOps.hessian' f >> fun (a, b) -> (a, Matrix.ofArray2D b)
    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f x = Vector.toArray >> DiffOps.hessian f >> Matrix.ofArray2D
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f x = Vector.toArray >> DiffOps.gradhessian' f >> fun (a, b, c) -> (a, vector b, Matrix.ofArray2D c)
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f x = Vector.toArray >> DiffOps.gradhessian f >> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x = Vector.toArray >> DiffOps.curl' f  >> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x = Vector.toArray >> DiffOps.curl f >> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x = Vector.toArray >> DiffOps.div' f >> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x = Vector.toArray >> DiffOps.div f
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x = Vector.toArray >> DiffOps.curldiv' f >> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x = Vector.toArray >> DiffOps.curldiv f >> fun (a, b) -> (vector a, b)
