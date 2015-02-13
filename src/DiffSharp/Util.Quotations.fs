//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
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

/// Utility functions used for handling quotations
module DiffSharp.Util.Quotations

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape


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

/// Get MethodInfo of an operation quoted in Expr `op`
let methInf (op:Expr) =
    match op with
    | Call(None, mi, _) -> mi
    | _ -> failwith "Unsupported operation."