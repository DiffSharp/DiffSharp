namespace rec DiffSharp.ShapeChecking

open System
open System.Collections.Concurrent
open System.Collections.Generic
open DiffSharp.ShapeChecking
open Microsoft.Z3

type SourceLocation = 
   { File: string
     StartLine: int 
     StartColumn: int 
     EndLine: int 
     EndColumn: int }
   override loc.ToString() = sprintf "%s: (%d,%d)-(%d,%d)" loc.File loc.StartLine loc.StartColumn loc.EndLine loc.EndColumn

type Diagnostic =
   { Severity: int
     Number: int
     Message: string
     LocationStack: SourceLocation[] }
   member x.Location = Array.last x.LocationStack

[<AutoOpen>]
module internal Util = 

    let (|Mul|_|) (s:Expr) = if s.IsMul then Some (s.Args) else None
    let (|IDiv|_|) (s:Expr) = if s.IsIDiv then Some (s.Args.[0], s.Args.[1]) else None
    let (|IntNum|_|) (s:Expr) = if s.IsIntNum then Some ((s :?> IntNum).Int) else None
    let (|Var|_|) (s:Expr) = if s.IsConst then Some s else None

    let rec isFreeIn (v: Expr) (tm: Expr) =
       if v = tm then true
       else if tm.IsApp then tm.Args |> Array.exists (fun arg -> isFreeIn v arg)
       else false

    let betterName (a: string) (b: string) = 
        b.StartsWith("?") && not (a.StartsWith("?"))

    let getEliminationMatrix (synAssertions: BoolExpr[]) (vars: Expr[]) (solver: Solver) =

        // Find the equations from the syntactic assertions and the normalised constrains
        let eqns = 
            [| for x in Array.append synAssertions solver.Assertions do 
                 if x.IsEq && x.Args.Length = 2 then 
                     yield x.Args.[0], x.Args.[1]

               // Each conseq is "true => v = const"
               for conseq in solver.Consequences([| |], vars) |> snd do
                   if conseq.Args.Length = 2 then 
                      let rhs = conseq.Args.[1]
                      yield rhs.Args.[0], rhs.Args.[1]
            |]

        // Find the equations defining variables, prefering a nicer name in a = b
        let veqns = 
            eqns 
            |> Array.choose (fun (a,b) -> 
                if a.IsConst && b.IsConst && betterName (a.ToString()) (b.ToString()) then Some(b,a) 
                elif a.IsConst then Some(a,b) 
                elif b.IsConst then Some(b,a) 
                else None)

        // Iteratively find all the equations where the rhs don't use any of the variables, e.g. "x = 1"
        // and normalise the e.h.s. of the other equations with respect to these
        let rec loop (veqns: (Expr * Expr)[]) acc =
            let relv, others = veqns |> Array.partition (fun (v,rhs) -> not (isFreeIn v rhs))
            match Array.toList relv with 
            | [] -> Array.ofList (List.rev acc)
            | (relv, rele)::others2 -> 
               let others = Array.append (Array.ofList others2) others
               let others = others |> Array.map (fun (v,b) -> (v.Substitute(relv, rele), b.Substitute(relv, rele)))
               loop others ((relv, rele) :: acc)
        loop veqns [ ]

[<RequireQualifiedAccess>]
type Sym(syms: SymScope, z3Expr: Expr) =

    member sym.SymScope = syms

    member sym.Z3Expr = z3Expr

    override sym.ToString() = sym.SymScope.Format(sym)

    interface ISym with 
      member sym.SymScope = (sym:Sym).SymScope :> ISymScope

[<AutoOpen>]
module SymbolPatterns =
    let (|Sym|) (x: ISym) : Sym = (x :?> Sym)

type SymVar(syms: SymScope, name: string, z3Expr: Expr) =
    member _.SymScope = syms
    member _.Name = name
    member _.Z3Expr = z3Expr
    override _.ToString() = "?" + name

type SymScope() =
    let zctx = new Context()
    let solver = zctx.MkSolver()
    let mutable elimCache = None
    //let zparams = zctx.MkParams()
    let mapping = ConcurrentDictionary<uint32, string * SourceLocation option>()
    let vars = ResizeArray<Expr>() // the variables 
    let synAssertions = ResizeArray<BoolExpr>() // the assertions made
    let diagnostics = ResizeArray<_>()
    let stack = Stack()

    member syms.Assert(func: string, args: Sym[]) =
        //printfn "constraint: %s(%s)" func (String.concat "," (Array.map string args))
        let expr = syms.Create(func, args)
        let zexpr = expr.Z3Expr :?> BoolExpr
        let res = solver.Check(zexpr)
        match res with
        | Status.UNSATISFIABLE ->
            false
        | _ -> 
            elimCache <- None
            solver.Assert(zexpr)
            synAssertions.Add(zexpr)
            true

    member syms.CreateFreshVar (name: string, ?location: SourceLocation) : Sym =
        let zsym = zctx.MkFreshConst(name, zctx.IntSort)
        mapping.[zsym.Id] <- (name, location)
        vars.Add(zsym)
        Sym (syms, zsym)

    member syms.CreateVar (name: string, ?location: SourceLocation) : Sym =
        let zbytes = System.Text.Encoding.UTF7.GetBytes(name)
        let zname = String(Array.map char zbytes)
        let zsym = zctx.MkConst(zname, zctx.IntSort)
        mapping.[zsym.Id] <- (name, location)
        vars.Add(zsym)
        Sym (syms, zsym)

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    member syms.CreateConst (v: obj) : Sym =
        let zsym = 
            match v with 
            | :? int as n -> zctx.MkInt(n) :> Expr
            | :? string as s -> zctx.MkString(s) :> Expr
            | _ -> failwithf "unknown constant %O or type %A" v (v.GetType())
        Sym(syms, zsym)

    interface ISymScope with
    
        override _.TryGetConst(sym) =
          match (sym :?> Sym).Z3Expr with 
          | IntNum n -> ValueSome (box n)
          | _ -> ValueNone

        override syms.CreateConst (v: obj) : ISym = syms.CreateConst (v) :> ISym

        override syms.CreateApp (f: string, args: ISym[]) : ISym =
            let args = args |> Array.map (fun (Sym x) -> x)
            syms.Create(f, args) :> ISym

        override syms.CreateFreshVar (name: string, location) : ISym =
            let loc = 
                match location with 
                | :? SourceLocation as loc ->  loc
                | _ -> { File = "?"; StartLine = 0; StartColumn = 0; EndLine = 0; EndColumn= 80 }
            syms.CreateFreshVar (name, loc) :> ISym

        override syms.AssertConstraint(func: string, args: ISym[]) =
            let args = args |> Array.map (fun (Sym x) -> x)
            syms.Assert(func, args)

        override _.ReportDiagnostic(severity, message) = diagnostics.Add((severity, message))

    member _.Push() = 
       stack.Push (synAssertions.ToArray())
       solver.Push()

    member _.Pop() = 
        let synAssertionsL = stack.Pop ()
        synAssertions.Clear(); for a in synAssertionsL do synAssertions.Add(a) 
        solver.Pop()

    member _.Clear() = solver.Reset()
    member syms.Create (f: string, args: Sym[]) : Sym =
        let zsym = 
           match f with 
           | "add" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkAdd(zargs) :> Expr
           | "mul" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkMul(zargs) :> Expr
           | "sub" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkSub(zargs) :> Expr
           | "div" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkDiv(zargs.[0], zargs.[1]) :> Expr
           | "mod" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> IntExpr)
               zctx.MkMod(zargs.[0], zargs.[1]) :> Expr
           | "leq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkLe(zargs.[0], zargs.[1]) :> Expr
           | "lt" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkLt(zargs.[0], zargs.[1]) :> Expr
           | "geq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkGe(zargs.[0], zargs.[1]) :> Expr
           | "gt" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkGt(zargs.[0], zargs.[1]) :> Expr
           | "eq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkEq(zargs.[0], zargs.[1]) :> Expr
           | "neq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkNot (zctx.MkEq(zargs.[0], zargs.[1])) :> Expr
           | s -> 
               //printfn "assuming %s is uninterpreted" s
               // TODO: string sorts and others
               let funcDecl = zctx.MkFuncDecl(s,[| for _x in args -> zctx.IntSort :> Sort |], (zctx.IntSort :> Sort))
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkApp(funcDecl, zargs)
       
        Sym(syms, zsym)

    /// Check the expression can be asserted without contradiction in the given context
    member syms.Check(func: string, args: Sym[]) : bool =
        let expr = syms.Create(func, args)
        let zexpr = expr.Z3Expr :?> BoolExpr
        let res = solver.Check(zexpr)
        res <> Status.UNSATISFIABLE

    /// Check if the given expression is always false in the given context
    member _.CheckAlwaysFalse(zexpr: Expr) : bool =
        let res = solver.Check(zexpr)
        res = Status.UNSATISFIABLE

    /// Check if the given expression is always true in the given context
    member _.CheckAlwaysTrue(zexpr: Expr) : bool =
        let res = solver.Check(zctx.MkNot(zexpr :?> BoolExpr))
        res = Status.UNSATISFIABLE

    member _.Solver = solver

    member _.GetAndClearDiagnostics() = 
        let res = diagnostics.ToArray()
        diagnostics.Clear()
        res

    /// Get a matrix eliminating variables 
    member _.GetEliminationMatrix () =
        let elim = 
            match elimCache with 
            | None -> 
                let res = getEliminationMatrix (synAssertions.ToArray()) (vars.ToArray()) solver 
                elimCache <- Some res
                res
            | Some e -> e
        elim

    /// Canonicalise an expression w.r.t. the equations in Solver
    member syms.EliminateVarEquations (expr: Expr) =
        let elim = syms.GetEliminationMatrix()
        expr.Substitute(Array.map fst elim, Array.map snd elim)

    member syms.GetAdditionalDiagnostics() : (int * SourceLocation option * string)[] =
        let elim = syms.GetEliminationMatrix()
        [| for (vsym, vexpr) in elim do
                match mapping.TryGetValue vsym.Id with
                | true, (vname, loc) -> 
                    let rhs = syms.Format(vexpr)
                    if not (vname.StartsWith("?")) then
                        (2, loc, sprintf "The symbol '%s' was constrained to be equal to '%s'" vname rhs)
                | _ -> ()
        |]

    member _.GetVarName(sym: Sym) = 
        match mapping.TryGetValue sym.Z3Expr.Id with
        | true, (v, _) -> v
        | _ -> "?"

    member _.GetVarLocation(sym: Sym) = 
        match mapping.TryGetValue sym.Z3Expr.Id with
        | true, (_, loc) -> Some loc
        | _ -> None

    member syms.Format(sym: Sym) = syms.Format(sym.Z3Expr)

    member syms.Format(zexpr: Expr) = 
        let parenIf c s = if c then "(" + s + ")" else s
        let isNegSummand (s: Expr) =
           match s with 
           | Mul [| IntNum n; _arg|] when n < 0 -> true
           | IntNum n when n < 0 -> true
           | _ -> false
        let rec print prec (zsym: Expr) =
            if zsym.IsAdd then
                // put negative summands at the end
                let args = zsym.Args |> Array.sortBy (fun arg -> if isNegSummand arg then 1 else 0)
                let argsText =
                   args 
                   |> Array.mapi (fun i arg -> 
                       match arg with 
                       | IntNum n when n < 0 -> string n
                       | Mul [| IntNum -1; arg|] -> "-"+print 2 arg
                       | Mul [| IntNum n; arg|] when n < 0 -> "-"+print 2 (zctx.MkMul(zctx.MkInt(-n),(arg :?> ArithExpr)))
                       | _ -> (if i = 0 then "" else "+") + print 2 arg)
                   |> String.concat ""
                parenIf (prec>1) argsText 
            elif zsym.IsSub then
                parenIf (prec>1) (zsym.Args |> Array.map (print 2) |> String.concat "-")
            elif zsym.IsMul then 
                parenIf (prec>4) (zsym.Args |> Array.map (print 4) |> String.concat "*")
            elif zsym.IsIDiv then 
                parenIf (prec>3) (zsym.Args |> Array.map (print 3) |> String.concat "/")
            elif zsym.IsRemainder then 
                parenIf (prec>3) (zsym.Args |> Array.map (print 3) |> String.concat "%")
            // simplify conditionals
            elif zsym.IsApp && zsym.FuncDecl.Name.ToString() = "if" && zsym.Args.Length = 3 && syms.CheckAlwaysTrue(zsym.Args.[0]) then
                //printfn "simplifying to then"
                print prec zsym.Args.[1]
            elif zsym.IsApp && zsym.FuncDecl.Name.ToString() = "if" && zsym.Args.Length = 3 && syms.CheckAlwaysFalse(zsym.Args.[0]) then
                //printfn "simplifying to else"
                print prec zsym.Args.[2]
            elif zsym.IsApp && zsym.Args.Length > 0 then 
                parenIf (prec>6) (zsym.FuncDecl.Name.ToString() + "(" + (zsym.Args |> Array.map (print 0) |> String.concat ",") + ")")
            elif zsym.IsConst then 
                match mapping.TryGetValue zsym.Id with
                | true, (v, _) -> v
                | _ -> zsym.ToString()
            else zsym.ToString()
        //printfn "pre-simplify %O" sym.Z3Expr
        let simp = zexpr.Simplify()
        //printfn "post-simplify %O" simp
        let simp2 = simp |> syms.EliminateVarEquations
        //printfn "post-elim%O" simp2
        print 0 simp2

