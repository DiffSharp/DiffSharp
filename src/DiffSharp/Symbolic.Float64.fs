// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

/// Symbolic differentiation module
module DiffSharp.Symbolic.Float64

open FSharp.Quotations
open FSharp.Quotations.Patterns
open FSharp.Quotations.DerivedPatterns
open FSharp.Quotations.ExprShape
open FSharp.Quotations.Evaluator
open DiffSharp.Util
open DiffSharp.Config

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
        | SpecificCall <@ log10 @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / (%%a * log10ValFloat64) @@>
        | SpecificCall <@ exp @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * %%expr @@>
        | SpecificCall <@ sin @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * cos %%a @@>
        | SpecificCall <@ cos @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float) * sin %%a @@>
        | SpecificCall <@ tan @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / ((cos %%a) ** 2.) @@>
        | SpecificCall <@ sqrt @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / (2. * %%expr) @@>
        | SpecificCall <@ sinh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * cosh %%a @@>
        | SpecificCall <@ cosh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) * sinh %%a @@>
        | SpecificCall <@ tanh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float) / ((cosh %%a) ** 2.) @@>
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
        fun x -> fj x |> GlobalConfig.Float64Backend.Transpose_M

    /// Original value and Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian' f =
        let fj = jacobianT' f
        fun x -> fj x |> fun (r, j) -> (r, GlobalConfig.Float64Backend.Transpose_M j)

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
        fun x -> fgh x |> drop1Of3

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then ErrorMessages.InvalidArgDiv()
        v, GlobalConfig.Float64Backend.Sum_V(GlobalConfig.Float64Backend.Diagonal_M(j))

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        let j = jacobianT f x
        if Array2D.length1 j <> Array2D.length2 j then ErrorMessages.InvalidArgDiv()
        GlobalConfig.Float64Backend.Sum_V(GlobalConfig.Float64Backend.Diagonal_M(j))

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], j.[0, 0] + j.[1, 1] + j.[2, 2]

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], j.[0, 0] + j.[1, 1] + j.[2, 2]
