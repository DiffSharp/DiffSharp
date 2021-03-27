namespace DiffSharp

open DiffSharp.Util

#if !NO_SYMBOLIC_SHAPES
open DiffSharp.ShapeChecking
    
/// Represents the shape of a tensor.  Each dimension may be symbolic.
[<Struct; CustomEquality; NoComparison>]
[<Symbolic>]
type Shape internal (values: int[], dims: Int[]) = 

    static member inline internal unop (x: Shape) f1 f2 =
        match x.TryGetValues() with 
        | ValueSome xv -> f1 xv
        | ValueNone -> f2 x.Dims

    static member inline internal binop (x: Shape) (y: Shape) f1 f2 =
        match x.TryGetValues(), y.TryGetValues() with 
        | ValueSome xv, ValueSome yv -> f1 xv yv
        | _, _ -> f2 x.Dims y.Dims

    /// Creates a constant shape from an array of integers
    new (values: seq<int>) = 
        let arr = Seq.toArrayQuick values
        for d in arr do
            if (d < -1) then failwithf "The shape dimension '%O' is less than -1. Shape dimensions must be positive, or else the indicator -1." d
        Shape(arr, null)

    /// Creates a possibly-symbolic shape from an array of possibly-symbolic integers
    new (dims: seq<Int>) =
        // assert all are either syntactically -1 placeholders or constrained > 0
        let dims = Seq.toArrayQuick dims
        for d in dims do
            if not (d = Int -1 || d >~ 0I) then failwithf "The shape dimension '%O' is zero or negative. Shape dimensions must be positive, or else the indicator -1." d
        let vs = dims |> Array.map (fun dim -> dim.TryGetValue())
        if vs |> Array.forall (fun v -> v.IsSome) then
            let values = vs |> Array.map (fun v -> v.Value)
            Shape(values, null)
        else 
            Shape(null, dims)

    /// Get the number of dimensions in the shape
    member _.Length =
        match values with 
        | null -> dims.Length
        | _ -> values.Length

    /// Get the possibly-symbolic dimensions of the shape
    member _.Dims =
        match values with 
        | null -> dims
        | _ -> values |> Array.map Int

    /// <summary>Get the values of the shape. Raises an exception if any of the dimensions are symbolic.</summary>
    /// <remarks>Symbolic dimensions will only appear when Backend.ShapeChecking is used.</remarks>
    member _.TryGetValues() =
        match values with 
        | null ->
            let vs = dims |> Array.map (fun dim -> dim.TryGetValue())
            if vs |> Array.forall (fun v -> v.IsSome) then
                ValueSome (vs |> Array.map (fun v -> v.Value))
            else ValueNone
        | _ -> ValueSome values

    /// <summary>Try to get a SymScope associated with a symbolic shape.</summary>
    /// <remarks>Symbolic shapes and dimensions will only appear when Backend.ShapeChecking is used.</remarks>
    member _.TryGetSymScope() =
        match values with 
        | null -> dims |> Array.tryPick (fun dim -> dim.TryGetSymScope())
        | _ -> None

    /// <summary>Get the values of the shape. Raises an exception if any of the dimensions are symbolic.</summary>
    /// <remarks>Symbolic dimensions will only appear when Backend.ShapeChecking is used.</remarks>
    member shape.Values =
        match shape.TryGetValues() with 
        | ValueSome values -> values
        | ValueNone -> failwithf "the shape '%A' is symbolic" shape

    /// <summary>Get a length of a particular dimension of the shape.</summary>
    /// <remarks>If the shape is symbolic then the length may be symbolic.</remarks>
    member _.Item with get i = 
        match values with 
        | null -> dims.[i]
        | _ -> Int values.[i]

    /// <summary>Gets the total number of elements in the shape.</summary>
    /// <remarks>
    ///   Raises an exception if any of the dimensions are symbolic. 
    ///   Symbolic dimensions will only appear when Backend.ShapeChecking is used.
    /// </remarks>
    member shape.nelement =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape.Values

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.nelementx =
        if shape.Length = 0 then Int 1
        else Array.reduce (*) shape.Dims

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.flatten() = 
        match values with 
        | null -> Shape [| shape.nelementx |]
        | _ -> Shape [| shape.nelement |]

    override x.Equals(y:obj) =
        match y with 
        | :? Shape as y ->
            match values, y.ValuesRaw with 
            | _, null | null, _ -> x.Dims = y.Dims
            | xvalues, yvalues -> xvalues = yvalues
        | _ -> false

    override shape.GetHashCode() = hash shape.Dims

    /// Constraint equality
    static member (=~=) (a:Shape,b:Shape) : bool = 
        Shape.binop a b (fun a b -> a = b) (fun a b -> a.Length = b.Length && (a,b) ||> Array.forall2(=~=))

    member _.GetSlice(low:int option,high:int option) =
        match values with 
        | null -> Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice dims low high)
        | _ -> Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice values low high)

    override x.ToString() = "[" + String.concat "," (x.Dims |> Array.map string) + "]"

    member internal _.ValuesRaw = values

    member internal _.DimsRaw = dims

    static member ParseSymbolic(env: Map<string, ISym>, syms: ISymScope, spec: obj, location: obj) : Shape =
        match spec with 
        | :? System.Reflection.ParameterInfo as p ->
            failwithf "%O: argument '%s' needs shape information in ShapeCheck attribute, e.g. [<ShapeCheck([| 1;4;2 |])>] or [<ShapeCheck([| \"N\";\"M\" |])>]  or [<ShapeCheck([| \"N\";\"M\" |])>]  or [<ShapeCheck([| \"N,M\" |])>] " location p.Name
        | :? int as n -> Shape [| n |]
        | :? (obj[]) as specs -> Shape [| for spec2 in specs -> Int.ParseSymbolic(env, syms, spec2, location) |]
        | :? string as text ->
            let parser = SymbolParser(env, syms, location)
            let toks = parser.TryParseShapeExpr(text)
            match toks with 
            | Some (es, true) -> Shape (Array.map Int.FromSymbol es)
            | _ -> failwithf "%O: invalid shape %s" location text
        | specObj -> failwithf "%O: invalid type for shape specification %s" location (specObj.GetType().ToString())

    member this.ConstrainSymbolic(other: Shape) : unit =
        if not (this =~= other) then
            failwithf "Shape mismatch. Expected '%O' but got '%O'" other this

#else

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

