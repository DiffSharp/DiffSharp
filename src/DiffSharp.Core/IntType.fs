namespace DiffSharp

open DiffSharp.ShapeChecking
open DiffSharp.Util

#if !NO_SYMBOLIC_SHAPES

/// <summary>
///  Represents an integer that may be symbolic, e.g. the size of one dimension of a tensor,
///  or an index into a tensor.
/// </summary>
///
/// <remarks>
///  Note that symbolic integers only appear when using Backend.ShapeChecking.  Otherwise
///  it can always be assumed that the symbol is empty.
/// </remarks>
[<Struct; CustomEquality; CustomComparison>]
type Int internal (n: int, sym: ISym) = 

    static member inline internal unop (x: Int) f1 f2 =
        match x.TryGetValue() with 
        | ValueSome xv -> f1 xv
        | ValueNone -> f2 x.SymbolRaw

    static member inline internal binop (x: Int) (y: Int) f1 f2 =
        match x.TryGetValue(), y.TryGetValue() with 
        | ValueSome xv, ValueSome yv -> f1 xv yv
        | ValueNone, ValueNone -> f2 x.SymbolRaw y.SymbolRaw
        | ValueSome _, ValueNone ->
            let symX = x.AsSymbol(y.SymbolRaw.SymScope)
            f2 symX y.SymbolRaw
        | ValueNone, ValueSome _ ->
            let symY = y.AsSymbol(x.SymbolRaw.SymScope)
            f2 x.SymbolRaw symY

    new (n: int) =
        //if n >= INTSYM_MIN && n <= INTSYM_MAX then
        //    match IntegerSyms.GetOrCreateIntegerSym(n) with
        //    | Some sym -> Int(0, sym)
        //    | None -> Int(n, Unchecked.defaultof<_>)
        //else 
        Int(n, Unchecked.defaultof<_>)

    static member FromSymbol (sym: ISym) = Int(0, sym)

    member internal x.SymbolRaw : ISym = sym

    member x.AsSymbol(syms: ISymScope) =
        match box sym with 
        | null  -> syms.CreateConst(n)
        | _ -> sym

    member x.IsSymbolic =
        match x.TryGetValue() with 
        | ValueSome _ -> false
        | ValueNone -> true

    member x.TryGetName() =
        match box sym with 
        | null -> ValueNone
        | _ -> ValueSome (sym.ToString())

    member x.TryGetValue() =
        match box sym with 
        | null -> ValueSome n
        | _ ->
            match sym.SymScope.TryGetConst(sym) with 
            | ValueSome (:? int as n) -> ValueSome n
            | _ -> ValueNone

    /// <summary>Try to get a SymScope associated with a symbolic integer.</summary>
    /// <remarks>Symbolic integers will only appear when Backend.ShapeChecking is used.</remarks>
    member _.TryGetSymScope() =
        match box sym with 
        | null -> None
        | _ -> Some sym.SymScope

    /// Return the value, exception if symbolic
    member x.Value =
        match x.TryGetValue() with 
        | ValueNone -> 
            failwithf """A construct required the value of a symbolic integer expression %s. Consider either 

- Changing your ShapeCheck to use concrete inputs, rather than symbolic, or
- Adjust the construct to propagate symbolic information, or
- Adjust your model to avoid dynamic dependencies on model inputs, or
- Add a check for symbolic tensor shapes, e.g. 'if tensor.symbolic then <return-dummy-tensor> else <main-code>'

Call stack: %A""" (sym.ToString()) (System.Diagnostics.StackTrace(fNeedFileInfo=true).ToString())
        | ValueSome v -> v

    /// Return the value, or '1' if this has no definite solution, normally to get a representative value
    member x.ValueOrOne =
        match x.TryGetValue() with 
        | ValueNone -> 1
        | ValueSome v -> v

    static member Max (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (max a b)) (fun a b -> Int.FromSymbol (ISym.binop "max" a b))

    static member Min (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (min a b)) (fun a b -> Int.FromSymbol (ISym.binop "min" a b))

    static member (+) (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (a+b)) (fun a b -> Int.FromSymbol (ISym.binop "add" a b))

    static member (+) (a:Int, b:int) : Int = a + Int b

    static member (+) (a:int, b:Int) : Int = Int a + b

    static member (-) (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (a-b)) (fun a b -> Int.FromSymbol (ISym.binop "sub" a b))

    static member (-) (a:Int, b:int) : Int = a - Int b

    static member (-) (a:int, b:Int) : Int = Int a - b

    static member (%) (a:Int,b:Int) : Int =
        Int.binop a b (fun a b -> Int (a%b)) (fun a b -> Int.FromSymbol (ISym.binop "mod" a b))

    static member (%) (a:Int, b:int) : Int = a % Int b

    static member (%) (a:int, b:Int) : Int = Int a % b

    static member (*) (a:Int,b:Int) : Int = 
        Int.binop a b (fun a b -> Int (a*b)) (fun a b -> Int.FromSymbol (ISym.binop "mul" a b))

    static member (*) (a:Int, b:int) : Int = a * Int b

    static member (*) (a:int, b:Int) : Int = Int a * b

    static member (/) (a:Int,b:Int) : Int = 
        Int.binop a b (fun a b -> Int (a/b)) (fun a b -> Int.FromSymbol (ISym.binop "div" a b))

    static member (/) (a:Int, b:int) : Int = a / Int b

    static member (/) (a:int, b:Int) : Int = Int a / b

    /// Negation operator
    static member (~-) (a:Int) : Int = 
        Int.unop a (fun a -> Int (-a)) (fun a -> Int.FromSymbol (ISym.unop "neg" a))

    /// Constraint equality
    static member (=~=) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a = b) (fun a b -> a.AssertEqualityConstraint(b))

    /// Constraint less-than-or-equal. Returns true if no contradiciton was detected when the constraint was asserted.
    static member (<=~) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a <= b) (fun a b -> a.SymScope.AssertConstraint("leq", [|a;b|]))

    /// Constraint less-than. Returns true if no contradiciton was detected when the constraint was asserted.
    static member (<~) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a < b) (fun a b -> a.SymScope.AssertConstraint("lt", [|a;b|]))

    /// Constraint greater-than. Returns true if no contradiciton was detected when the constraint was asserted.
    static member (>~) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a > b) (fun a b -> a.SymScope.AssertConstraint("gt", [|a;b|]))

    /// Constraint greater-than. Returns true if no contradiciton was detected when the constraint was asserted.
    static member (>=~) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a >= b) (fun a b -> a.SymScope.AssertConstraint("geq", [|a;b|]))

    static member (<~) (a:Int,b:int) : bool = a <~ Int b
    static member (<=~) (a:Int,b:int) : bool = a <=~ Int b
    static member (>=~) (a:Int,b:int) : bool = a >=~ Int b
    static member (>~) (a:Int,b:int) : bool = a >~ Int b

    static member Zero = Int 0

    static member Abs(dim: Int) = Int (abs dim.Value)

    member _.IsUnspecified = (n = -1)

    member _.IsInvalid = (n < -1)

    override x.GetHashCode() =
        match x.TryGetValue() with 
        | ValueNone -> 0
        | ValueSome v -> v

    override x.Equals(y:obj) =
          match y with 
          | :? Int as y -> Int.binop x y (=) (fun xsym ysym -> obj.ReferenceEquals(xsym,ysym))
          | _ -> false

    interface System.IComparable with 
       member x.CompareTo(y:obj) = 
          match y with 
          | :? Int as y -> compare x.Value y.Value // TODO - symbols
          | _ -> failwith "wrong type"

    override x.ToString() =
        match x.TryGetValue() with 
        | ValueNone -> string sym
        | ValueSome v -> string v

#else

/// Represents an integer used in as a dimension, e.g. the size of one dimension of a tensor,
/// or an index into a tensor.
[<Struct; CustomEquality; CustomComparison>]
type Int (n: int) = 

    /// Return the value
    member x.Value = n

    /// Return the value
    member x.ValueOrOne = n

    static member (+) (a:Int, b:Int) : Int = Int (a.Value + b.Value)

    static member (+) (a:Int, b:int) : Int = Int (a.Value + b)

    static member (-) (a:Int, b:Int) : Int = Int (a.Value - b.Value)

    static member (-) (a:Int, b:int) : Int = Int (a.Value - b)

    static member (%) (a:Int,b:Int) : Int = Int (a.Value % b.Value)

    static member (*) (a:Int,b:Int) : Int = Int (a.Value * b.Value)

    static member (*) (a:int,b:Int) : Int = Int (a * b.Value)

    static member (*) (a:Int,b:int) : Int = Int (a.Value * b)

    static member (/) (a:Int,b:Int) : Int = Int (a.Value / b.Value)

    static member (/) (a:Int,b:int) : Int = Int (a.Value / b)

    static member Zero = Int 0

    static member Abs(dim: Int) = Int (abs dim.Value)

    member _.IsUnspecified = (n = -1)

    member _.IsInvalid = (n < -1)

    override x.GetHashCode() = n

    override x.Equals(y:obj) =
          match y with 
          | :? Int as y ->  n = y.Value
          | _ -> false

    interface System.IComparable with 
       member x.CompareTo(y:obj) = 
          match y with 
          | :? Int as y -> compare x.Value y.Value
          | _ -> failwith "wrong type"

    override x.ToString() = string x.Value

    /// Constraint equality
    static member (=~=) (a:Int,b:Int) : bool = (a.Value = b.Value)

    /// Constraint less-than-or-equal. 
    static member (<=~) (a:Int,b:Int) : bool = (a.Value <= b.Value)

/// Represents the shape of a tensor.
[<Struct; CustomEquality; NoComparison>]
type Shape (values: int[]) = 

    new (values: Int[]) = 
        let valuesi : int[] = values |> Array.map (fun v -> v.Value)
        Shape (valuesi)

    /// Get the number of dimensions in the shape
    member _.Length = values.Length

    member internal _.ValuesRaw = values

    /// Get the possibly-symbolic dimensions of the shape
    member _.Dims = Array.map Int values

    member _.Values = values

    member _.Item with get i = Int values.[i]

    /// <summary>Gets the total number of elements in the shape.</summary>
    member shape.nelement =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape.Values

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.nelementx = Int shape.nelement 

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.flatten() = Shape [| shape.nelement |]

    override x.Equals(y:obj) =
        match y with 
        | :? Shape as y -> values = y.ValuesRaw
        | _ -> false

    override x.GetHashCode() = hash x.Dims

    member _.GetSlice(x:int option,y:int option) =
        Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice values x y)

    override x.ToString() = "[" + String.concat "," (x.Values |> Array.map string) + "]"

    /// Constraint equality
    static member (=~=) (a:Shape,b:Shape) : bool =  (a.Values = b.Values)

#endif

// When using shape checking the syntax 128I is hijacked
module NumericLiteralI = 
    let FromZero () : Int = Int 0
    let FromOne () : Int = Int 1
    let FromInt32 (value:int32): Int = Int value

