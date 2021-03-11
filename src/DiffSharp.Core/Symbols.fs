namespace DiffSharp.ShapeChecking

open System

type ISym =
    abstract SymScope : ISymScope

/// Represents an accumulating collection of related symbols and constraints
and ISymScope =

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    abstract CreateConst: v: obj -> ISym 

    /// Create an application symbol
    abstract CreateApp: func: string * args: ISym[] -> ISym 

    /// Create a variable symbol. If fresh is true it is distinct from any other symbol of the same type in this scope,
    /// attaching the given additional information to the variable, e.g. a location
    abstract CreateVar: name: string * location: obj * ?fresh: bool -> ISym

    /// Try to get the symbol as a constant
    abstract TryGetConst: ISym -> obj voption

    /// Asserts a constraint in the solver state, returning true if the constraint is consistent
    /// with the solver state, and false if it is inconsistent.
    abstract AssertConstraint: func: string * args: ISym[]  -> bool

    /// Report a diagnostic related to this set of symbols and their constraints.
    /// Severity is 0=Informational, 1=Warning, 2=Error.
    abstract ReportDiagnostic: severity: int * message: string -> unit

[<AttributeUsage(AttributeTargets.Class ||| AttributeTargets.Interface ||| AttributeTargets.Struct ||| AttributeTargets.Delegate)>]
type SymbolicAttribute() = inherit System.Attribute()

//type IIntegerSyms =
//    abstract GetOrCreateIntegerSym: int -> ISym option

[<AutoOpen>]
module SymbolExtensions =

    //// This is the handle for embedding integer symbols via integers
    //let mutable IntegerSyms =
    //    { new IIntegerSyms with
    //        member _.GetOrCreateIntegerSym(n:int) = failwith "no integer sym handler installed" }

    type ISym with   

        static member unop nm (arg: ISym) : ISym =
            arg.SymScope.CreateApp(nm, [|arg|])

        static member binop nm (arg1: ISym) (arg2: ISym) : ISym =
            arg1.SymScope.CreateApp(nm, [|arg1; arg2|])

        static member app nm (args: ISym []) : ISym =
            args.[0].SymScope.CreateApp(nm, args)

        /// Assert the two symbols to be equal
        member sym1.AssertEqualityConstraint(sym2) =
            sym1.SymScope.AssertConstraint("eq", [|sym1; sym2|])

    //// "249421277 should be enough for anyone"
    //let [<Literal>] INTSYM_MIN = 0x71111111 
    //let [<Literal>] INTSYM_MAX = 0x7FEEEEEE 

type SymbolParser(env: Map<string, ISym>, syms: ISymScope, loc: obj) =
    let (|Integer|_|) toks = 
        match toks with 
        | Choice1Of3 c :: rest -> Some (c, rest)
        | _ -> None
    let (|Ident|_|) toks = 
        match toks with 
        | Choice3Of3 c :: rest -> Some (c, rest)
        | _ -> None
    let (|Symbol|_|) toks = 
        match toks with 
        | Choice2Of3 c :: rest -> Some (c, rest)
        | _ -> None
    //printfn "making symbolic for model parameter %s, givenArgInfo = %A" p.Name givenArgInfo
    let isSymbolChar c =
            c = '+' || 
            c = '/' || 
            c = '*' || 
            c = '×' || // OK, I like unicode
            c = '-' || 
            c = '[' || 
            c = ']' || 
            c = ',' || 
            c = '(' || 
            c = ')' 
    let tokenize (text: string) = 
        [ let mutable i = 0 
          while i < text.Length do
                if Char.IsDigit (text, i) then
                    let start = i
                    while i < text.Length && (Char.IsDigit (text, i)) do
                        i <- i + 1
                    yield Choice1Of3 (Int32.Parse text.[start..i-1])
                elif text.[i] = '+' || 
                     text.[i] = '/' || 
                     text.[i] = '*' || 
                     text.[i] = '×' || // OK, I like unicode
                     text.[i] = '-' || 
                     text.[i] = '[' || 
                     text.[i] = ']' || 
                     text.[i] = ',' || 
                     text.[i] = '(' || 
                     text.[i] = ')' then
                    let tok = text.[i..i]
                    i <- i + tok.Length
                    yield Choice2Of3 tok
                elif Char.IsLetter (text, i) || (Char.IsSymbol (text, i) && not (isSymbolChar text.[i])) then
                    let start = i
                    while i < text.Length  && (Char.IsLetter (text, i) || (Char.IsSymbol (text, i) && not (isSymbolChar text.[i])) || Char.IsDigit (text, i)) do
                        if Char.IsSurrogatePair(text, i) then 
                            i <- i + 2
                        else
                            i <- i + 1
                    yield Choice3Of3 text.[start..i-1]
                elif Char.IsWhiteSpace (text, i) then
                    i <- i + 1
                else  
                    failwithf "%O: unknown character '%c' in expression" loc text.[i] ]
    let rec (|IntExpr|_|) toks = 
        match toks with 
        | DivExpr (e1, Symbol("+", IntExpr (e2, rest))) -> Some (syms.CreateApp("add", [| e1; e2 |]), rest)
        | DivExpr (e1, Symbol("-", IntExpr (e2, rest))) -> Some (syms.CreateApp("sub", [| e1; e2 |]), rest)
        | DivExpr (e, rest) -> Some (e, rest)
        | _ -> None
    and (|DivExpr|_|) toks = 
        match toks with 
        | MulExpr (e1, Symbol("/", MulExpr (e2, rest))) -> Some (syms.CreateApp("div", [| e1; e2 |]), rest)
        | MulExpr (e, rest) -> Some (e, rest)
        | _ -> None
    and (|MulExpr|_|) toks = 
        match toks with 
        | AtomExpr (e1, Symbol("*", MulExpr (e2, rest))) -> Some (syms.CreateApp("mul", [| e1; e2 |]), rest)
        | AtomExpr (e, rest) -> Some (e, rest)
        | _ -> None
    and (|AtomExpr|_|) toks = 
        match toks with 
        | Symbol ("(", IntExpr (e, Symbol (")", rest))) -> Some (e, rest)
        | Integer (n, rest) -> Some (syms.CreateConst n, rest)
        | Ident (n, rest) -> 
            match env.TryFind n with 
            | Some sym -> Some (sym, rest)
            | None -> Some (syms.CreateVar (n, loc), rest)
        | _ -> None
    and (|IntsExprs|_|) toks = 
        match toks with 
        | IntExpr (e, Symbol (("," | "×"), IntsExprs (es, rest))) -> Some (e :: es, rest)
        | IntExpr (e, rest) -> Some ([e], rest)
        | _ -> None
    and (|ShapeExpr|_|) toks = 
        match toks with 
        | Symbol ("[", IntsExprs (es, Symbol ("]", rest))) -> Some (Array.ofList es, rest)
        | IntsExprs (es, rest) -> Some (Array.ofList es, rest)
        | _ -> None

    member _.TryParseIntExpr(text) = text |> tokenize |> (|IntExpr|_|) |> Option.map (fun (a,b) -> a, (b.Length = 0))
    member _.TryParseShapeExpr(text) = text |> tokenize |> (|ShapeExpr|_|) |> Option.map (fun (a,b) -> a, (b.Length = 0))

