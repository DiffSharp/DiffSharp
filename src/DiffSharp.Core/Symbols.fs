namespace DiffSharp.ShapeChecking

type ISym =
    abstract SymScope : ISymScope

/// Represents an accumulating collection of related symbols and constraints
and ISymScope =

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    abstract CreateConst: v: obj -> ISym 

    /// Create an application symbol
    abstract CreateApp: func: string * args: ISym[] -> ISym 

    /// Create a variable symbol, distinct from any other symbol of the same type in this scope,
    /// attaching the given additional information to the variable, e.g. a location
    abstract CreateFreshVar: name: string * location: obj -> ISym

    /// Try to get the symbol as a constant
    abstract TryGetConst: ISym -> obj voption

    /// Asserts a constraint in the solver state, returning true if the constraint is consistent
    /// with the solver state, and false if it is inconsistent.
    abstract AssertConstraint: func: string * args: ISym[]  -> bool

    /// Report a diagnostic related to this set of symbols and their constraints.
    /// Severity is 0=Informational, 1=Warning, 2=Error.
    abstract ReportDiagnostic: severity: int * message: string -> unit

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

