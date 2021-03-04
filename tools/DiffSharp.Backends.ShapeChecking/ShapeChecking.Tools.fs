namespace rec DiffSharp.ShapeChecking

open System
open System.Reflection
open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking
open DiffSharp.Backends.ShapeChecking
open System.Runtime.CompilerServices
open System.Collections.Concurrent

type ShapeCheckingReturnType =
    ((int (* severity *) * 
      string (* prefix *) * 
      int (* number *) * 
      (string * int * int * int * int)[] * (* stack locations *)
      (* message *) string)[])

[<AutoOpen>]
module ShapeCheckingAutoOpens =
    type SymScope with 
        member syms.CreateFreshIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateFreshVar(name, ?location=location))

        member syms.CreateIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateVar(name, ?location=location))

        /// Create an inferred symbol 
        member syms.Infer = syms.CreateFreshIntVar("?")

    /// Create a symbol in the global symbol context of the given name
    let (?) (syms: SymScope) (name: string) : Int = syms.CreateIntVar(name)

[<AutoOpen>]
module Tools =

    /// Record a stack of ranges in an exception. This uses exactly the same protocol as FsLive
    type System.Exception with 
        member e.EvalLocationStack 
            with get() = 
                if e.Data.Contains "location" then 
                    match e.Data.["location"] with 
                    | :? ((string * int * int * int * int)[]) as stack -> stack
                    | _ -> [| |]
                else
                    [| |]
            and set (data : (string * int * int * int * int)[]) = 
                e.Data.["location"] <- data

    let DiagnosticFromException  (loc: SourceLocation) (err: exn) =
        let stack = [| for (f,sl,sc,el,ec) in err.EvalLocationStack -> { File=f;StartLine=sl;StartColumn=sc;EndLine=el;EndColumn=ec }  |]
        { Severity=2; 
          Number = 1001
          Message = err.Message
          LocationStack = Array.append [| loc |] stack }

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

    type ParserLogic(env: Map<string, Int>, syms: SymScope, loc) =
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
        let rec (|Expr|_|) toks = 
            match toks with 
            | DivExpr (e1, Symbol("+", Expr (e2, rest))) -> Some (syms.Create("add", [| e1; e2 |]), rest)
            | DivExpr (e1, Symbol("-", Expr (e2, rest))) -> Some (syms.Create("sub", [| e1; e2 |]), rest)
            | DivExpr (e, rest) -> Some (e, rest)
            | _ -> None
        and (|DivExpr|_|) toks = 
            match toks with 
            | MulExpr (e1, Symbol("/", MulExpr (e2, rest))) -> Some (syms.Create("div", [| e1; e2 |]), rest)
            | MulExpr (e, rest) -> Some (e, rest)
            | _ -> None
        and (|MulExpr|_|) toks = 
            match toks with 
            | AtomExpr (e1, Symbol("*", MulExpr (e2, rest))) -> Some (syms.Create("mul", [| e1; e2 |]), rest)
            | AtomExpr (e, rest) -> Some (e, rest)
            | _ -> None
        and (|AtomExpr|_|) toks = 
            match toks with 
            | Symbol ("(", Expr (e, Symbol (")", rest))) -> Some (e, rest)
            | Integer (n, rest) -> Some (syms.CreateConst n, rest)
            | Ident (n, rest) -> 
                match env.TryFind n with 
                | Some sym -> Some ((sym.AsSymbol(syms) :?> Sym), rest)
                | None -> Some (syms.CreateVar (n, loc), rest)
            | _ -> None
        and (|IntExprs|_|) toks = 
            match toks with 
            | Expr (e, Symbol (("," | "×"), IntExprs (es, rest))) -> Some (e :: es, rest)
            | Expr (e, rest) -> Some ([e], rest)
            | _ -> None
        and (|ShapeExpr|_|) toks = 
            match toks with 
            | Symbol ("[", IntExprs (es, Symbol ("]", rest))) -> Some (Shape (Array.map Int.FromSymbol (Array.ofList es)), rest)
            | IntExprs (es, rest) -> Some (Shape (Array.map Int.FromSymbol (Array.ofList es)), rest)
            | _ -> None

        let parseSymbolicIntArg (spec: string) (loc: SourceLocation) : Int =
            let toks = tokenize spec
            let sym = 
                match toks with 
                | Expr (e, []) -> e
                | _ -> failwithf "%O: invalid expression %s" loc spec
            Int.FromSymbol sym

        let parseSymbolicShapeArg (spec: string) (loc: SourceLocation) : Shape =
            let toks = tokenize spec
            match toks with 
            | ShapeExpr (e, []) -> e
            | _ -> failwithf "%O: invalid shape %s" loc spec

        let getSymbolicIntArg (givenArgInfo: (obj * SourceLocation) option) (p: ParameterInfo) (loc: SourceLocation) : Int =
            //printfn "making symbolic for model parameter %s, givenArgInfo = %A" p.Name givenArgInfo
            match givenArgInfo with 
            | Some (:? int as n, _) -> Int n 
            | Some (:? string as text, loc) -> parseSymbolicIntArg text loc
            | Some (arg, loc) -> failwithf "%O: unknown specification %A for argument '%s'" loc arg p.Name
            | None -> syms.CreateIntVar(p.Name, loc) 

        let getSymbolicShapeArg (givenArgInfo: (obj * SourceLocation) option) (p: ParameterInfo) (loc: SourceLocation) : Shape =
            //printfn "making symbolic for model parameter %s" p.Name
            match givenArgInfo with 
            | Some (:? int as n, _) -> Shape [| n |]
            | Some (:? string as nm, loc) -> parseSymbolicShapeArg nm loc
            | Some (:? (obj[]) as specs, loc) -> 
                Shape [| for nm in specs -> 
                            match nm with 
                            | :? int as n -> Int n
                            | :? string as spec -> parseSymbolicIntArg spec loc 
                            | arg -> failwithf "%O: unknown specification %A for argument '%s'" loc arg p.Name |]
            | Some (arg, loc) -> failwithf "%O: unknown specification %A for argument '%s'" loc arg p.Name
            | None -> failwithf "%O: argument '%s' needs shape information in ShapeCheck attribute, e.g. [<ShapeCheck([| 1;4;2 |])>] or [<ShapeCheck([| \"N\";\"M\" |])>] " loc p.Name

        let getSymbolicTensorArg givenArgInfo (p: ParameterInfo) loc : Tensor =
            let shape = getSymbolicShapeArg givenArgInfo p loc
            dsharp.zeros(shape)

        let getSampleArg (givenArgInfo: (obj * SourceLocation) option) (p: ParameterInfo) (dflt: 'T) loc : 'T =
            //printfn "making symbolic for model parameter %s" p.Name
            match givenArgInfo with 
            | Some (:? 'T as n, _) -> n 
            | Some (:? string as spec, _) when typeof<'T> = typeof<int> -> 
                let s = parseSymbolicIntArg spec loc
                if s.IsSymbolic then
                    failwithf "%O: This shape check uses a symbol where a integer value is expected.  Either update your model to use the special 'Int' (capitalised) type for dimensions, indexes and shapes (TODO: give link to guide), or change to a specific integer value" loc
                s.Value |> box |> unbox
            | Some (arg, loc) -> failwithf "%O: unknown arg specification %A" loc arg
            | None -> 
                printfn "%O: assuming sample value '%O' for model parameter %s" loc dflt p.Name
                dflt

        member _.GetArg optionals givenArgInfo (p: ParameterInfo) loc : obj * Diagnostic[] =
            let pty = p.ParameterType
            let pts = pty.ToString()
            if pts = "DiffSharp.Int" then
                getSymbolicIntArg givenArgInfo p loc |> box, [||]
            elif pts = "DiffSharp.Shape" then
                getSymbolicShapeArg givenArgInfo p loc |> box, [||]
            elif pts = "DiffSharp.Tensor" then
                getSymbolicTensorArg givenArgInfo p loc |> box, [||]
            elif pts = "DiffSharp.ShapeChecking.ISymScope" || pts = "DiffSharp.ShapeChecking.SymScope" then
                syms |> box, [||]
            elif pts = "System.Int32" then 
                getSampleArg givenArgInfo p 1 loc |> box, [||]
            elif pts = "System.Single" then 
                getSampleArg givenArgInfo p 1.0f loc |> box, [||]
            elif pts = "System.Double" then 
                getSampleArg givenArgInfo p 1.0 loc |> box, [||]
            elif pts = "System.Boolean" then 
                getSampleArg givenArgInfo p true loc |> box, [||]
            elif pts = "System.String" then 
                getSampleArg givenArgInfo p "" loc|> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[DiffSharp.Int]" then 
                getSymbolicIntArg givenArgInfo p loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[DiffSharp.Shape]" then 
                getSymbolicShapeArg givenArgInfo p loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[DiffSharp.Tensor]" then 
                // Only lay down an optional tensor arg if ndims info has actually been given
                givenArgInfo |> Option.bind (fun _ -> getSymbolicTensorArg givenArgInfo p loc |> Some) |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[System.Boolean]" then 
                getSampleArg givenArgInfo p true loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[System.Int32]" then 
                getSampleArg givenArgInfo p 1 loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[System.Single]" then 
                getSampleArg givenArgInfo p 1.0f loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[System.Double]" then 
                getSampleArg givenArgInfo p 1.0 loc |> Some |> box, [||]
            elif optionals && pts = "Microsoft.FSharp.Core.FSharpOption`1[System.String]" then 
                getSampleArg givenArgInfo p "" loc |> Some |> box, [||]
            elif pts.StartsWith("Microsoft.FSharp.Core.FSharpOption`1[") then 
                let warns = 
                    if optionals then 
                        let msg = sprintf "Optional model parameter '%s' has unknown type '%O' for shape checking. A 'None' value will be assumed." p.Name p.ParameterType
                        [| { Severity=1; LocationStack=[| loc |]; Message=msg; Number=1999 } |]
                    else
                        [| |]
                null, warns
            else 
                let msg = sprintf "Model parameter '%s' has unknown type '%O' for shape checking. A 'null' value will be assumed. Consider changing the type or extending the shape checking tools to understand this type of argument" p.Name p.ParameterType
                null, [| { Severity=1; LocationStack=[| loc |]; Message=msg; Number=1999 } |]

        member t.GetParams optionals  (ps: ParameterInfo[]) (givenArgInfos: obj[]) loc =
            [| for i in 0 .. ps.Length - 1 do 
                    let p = ps.[i]
                    let givenArgInfo = (if i < givenArgInfos.Length then Some (givenArgInfos.[i], loc) else None)
                    p.Name, t.GetArg optionals givenArgInfo p loc |]

    // Constrain the return shape
    let constrainReturnValueByShapeInfo env syms (retActual: obj) (shapeInfo: obj) (retParam: ParameterInfo) loc : Result<unit, Diagnostic> * Diagnostic[] =
        match retActual, shapeInfo with 
        | null, _ | _, null ->  Ok (), [| |]
        | (:? Tensor | :? Shape | :? Int), info -> 
            let logic = ParserLogic(env, syms, loc) 
            let retReqd, warns = logic.GetArg false (Some (info, loc)) retParam loc 
            match retReqd, retActual with 
            | (:? Tensor as retReqd), (:? Tensor as retActual) ->
                if not (retActual.shapex =~= retReqd.shapex) then
                    let msg = sprintf "Shape mismatch. Expected a tensor with shape '%O' but got shape '%O'" retReqd.shapex retActual.shapex
                    Error { Severity=2; LocationStack=[| loc |]; Message=msg; Number=1999 }, warns
                else
                   Ok (), warns
            | (:? Shape as retReqd), (:? Shape as retActual) ->
                if not (retActual =~= retReqd) then
                    let msg = sprintf "Shape mismatch. Expected shape '%O' but got shape '%O'" retReqd retActual
                    Error { Severity=2; LocationStack=[| loc |]; Message=msg; Number=1999 }, warns
                else
                   Ok (), warns
            | (:? Int as retReqd), (:? Int as retActual) ->
                if not (retActual =~= retReqd) then
                    let msg = sprintf "Shape mismatch. Expected '%O' but got '%O'" retReqd retActual
                    Error { Severity=2; LocationStack=[| loc |]; Message=msg; Number=1999 }, warns
                else
                   Ok (), warns
            | _, _ -> 
                let msg = sprintf "Unknown return type for shape checking"
                Ok(), Array.append [| { Severity=1; LocationStack=[| loc |]; Message=msg; Number=1999 } |] warns
        | _ -> 
            let msg = sprintf "Unexpected return from method with ReturnShape attribute" 
            Error { Severity=1; LocationStack=[| loc |]; Message=msg; Number=1999 }, [| |]

    let invokeShapeCheckMeth (syms: SymScope) optionals env (meth: MethodInfo) (attr:  ShapeCheckAttribute) (file, sl, sc, el, ec) (model: obj) =
        let diags = ResizeArray()
        let methCalls = ResizeArray()
        if not meth.ContainsGenericParameters && not meth.DeclaringType.ContainsGenericParameters then
                // Use a better location for the method attribute if given
                let mloc = { File=file; StartLine=sl; StartColumn=sc; EndLine=el; EndColumn=ec} 
                      
                try 
                    syms.Push()
                    let args = ParserLogic(env, syms, mloc).GetParams optionals (meth.GetParameters()) attr.GivenArgs mloc

                    let argValues = 
                        [| for (_, (arg, warns)) in args do
                            diags.AddRange warns
                            arg |]

                    let res =
                        try 
                            meth.Invoke (model, argValues) |> Ok
                        with :?TargetInvocationException as e -> 
                            let e = e.InnerException
                            Error (DiagnosticFromException mloc e)

                    match res with
                    | Ok retActual -> 
                        let retOk, retWarns = constrainReturnValueByShapeInfo env syms retActual attr.ReturnShape meth.ReturnParameter mloc 
                        diags.AddRange retWarns
                        match retOk with
                        | Ok () -> ()
                        | Error diag -> diags.Add diag

                        // Show extra information about over-constrained variables
                        let moreDiags = syms.GetAdditionalDiagnostics()
                  
                        for (severity, loc2, msg) in moreDiags do   
                            let stack = Array.append (Option.toArray loc2) [| mloc |]
                            diags.Add ({ Severity=severity; LocationStack=stack; Message = msg; Number=1996 })
                        methCalls.Add(meth, args, Ok retActual)
                    | Error e -> 
                        methCalls.Add(meth, args, Error e)
                        diags.Add e
                finally
                    syms.Pop()
        diags.ToArray(), methCalls.ToArray()

    let makeModelAndRunShapeChecks (syms: SymScope) optionals (ctor: ConstructorInfo) ctorGivenArgs tloc subTargets =
        let diags = ResizeArray()
        let calls = ResizeArray()
        
        let ctorArgs = ParserLogic(Map.empty, syms, tloc).GetParams optionals (ctor.GetParameters()) ctorGivenArgs tloc
        let ctorArgValues = 
            [| for (nm, (arg, warns)) in ctorArgs do
                diags.AddRange warns
                (nm, arg) |]

        let env = 
            ctorArgValues 
            |> Array.choose (fun (nm, v) -> 
                    match v with
                    | :? Int as n -> Some (nm, n)
                    | :? int as n -> Some (nm, Int n)
                    | :? (int option) as n when n.IsSome -> Some (nm, Int n.Value)
                    | :? (Int option) as n when n.IsSome -> Some (nm, n.Value)
                    | _ -> None)
            |> Map.ofArray
        
        // Invoke the constructor to target the model
        let model = 
            try ctor.Invoke(Array.map snd ctorArgValues) |> Ok
            with :? TargetInvocationException as e -> Error (DiagnosticFromException tloc e.InnerException)

        match model with 
        | Error e ->
            ctorArgValues, Error e, [| |], [| e |]
        | Ok model ->

            // Invoke each shape check target in model
            for (subTargetMeth: MethodInfo, subTargetAttr: obj, subTargetLoc) in subTargets do
                match subTargetAttr with 
                | :? ShapeCheckAttribute as subTargetAttr ->
                    let methDiags, methCalls = invokeShapeCheckMeth syms optionals env subTargetMeth subTargetAttr subTargetLoc model
                    diags.AddRange(methDiags)
                    calls.AddRange(methCalls)
                | _ -> ()

            ctorArgValues, Ok (model), calls.ToArray(), diags.ToArray()

/// When added a to model or its methods, indicates that ShapeCheck tooling should analyse the shapes
/// of the construct.
[<AttributeUsage(validOn=AttributeTargets.All, AllowMultiple=true, Inherited=true)>]
type ShapeCheckAttribute internal (given: obj[]) =
    inherit System.Attribute()
    new () =
        ShapeCheckAttribute([| |] : obj[])
    new (argShape1: obj) =
        ShapeCheckAttribute([| argShape1 |])
    new (argShape1: obj, argShape2: obj) =
        ShapeCheckAttribute([| argShape1; argShape2 |])
    new (argShape1: obj, argShape2: obj, argShape3: obj) =
        ShapeCheckAttribute([| argShape1; argShape2; argShape3 |])
    new (argShape1: obj, argShape2: obj, argShape3: obj, argShape4: obj) =
        ShapeCheckAttribute([| argShape1; argShape2; argShape3; argShape4 |])
    new (argShape1: obj, argShape2: obj, argShape3: obj, argShape4: obj, argShape5: obj) =
        ShapeCheckAttribute([| argShape1; argShape2; argShape3; argShape4; argShape5 |])
    new (argShape1: obj, argShape2: obj, argShape3: obj, argShape4: obj, argShape5: obj, argShape6: obj) =
        ShapeCheckAttribute([| argShape1; argShape2; argShape3; argShape4; argShape5; argShape6 |])
    new (argShape1: obj, argShape2: obj, argShape3: obj, argShape4: obj, argShape5: obj, argShape6: obj, argShape7: obj) =
        ShapeCheckAttribute([| argShape1; argShape2; argShape3; argShape4; argShape5; argShape6; argShape7 |])

    member val ReturnShape : obj = null with get, set

    member _.GivenArgs = given

    /// 'fslive' invokes this member with the right information and expects exactly this goopo of information
    /// back
    ///
    /// TODO: see if there are standard types somewhere to use for this
    member attr.RunChecks(target: obj (* System.Type | System.MethodInfo *) ,
             loc: (string * int * int * int * int), 
             subTargets: (MethodInfo * obj * (string * int * int * int * int))[]) 
            : ShapeCheckingReturnType =

        let _ = System.Runtime.InteropServices.NativeLibrary.Load("libz3", System.Reflection.Assembly.GetExecutingAssembly(), Nullable())
        let optionals = true
        let (locFile, locStartLine, locStartColumn, locEndLine, locEndColumn) = loc
        
        let syms = SymScope()
        let diags =
            match target with 
            | :? System.Type as targetType -> 
                let ctors = targetType.GetConstructors()
                let ctor = 
                    ctors 
                    |> Array.tryFind (fun ctor -> 
                        ctor.GetParameters() |> Array.exists (fun p -> 
                            let pt = p.ParameterType.ToString()
                            pt.Contains("DiffSharp.Int") || pt.Contains("DiffSharp.Shape")))
                    |> function 
                       | None -> 
                           //printf "couldn't find a model constructor taking Int or Shape parameter, assuming first constructor is target of live check"
                           ctors.[0]
                       | Some c -> c

                let tloc = { File = locFile; StartLine = locStartLine; StartColumn = locStartColumn; EndLine = locEndLine; EndColumn= locEndColumn }
                let _, _, _, diags = makeModelAndRunShapeChecks syms optionals ctor attr.GivenArgs tloc subTargets
                diags
            | :? System.Reflection.MethodInfo as meth -> 
                let methDiags, _methCalls = invokeShapeCheckMeth syms optionals Map.empty meth attr loc null
                methDiags
            | _ -> 
                [| |]

        [| for diag in diags -> 
            let stack = 
                [| for m in diag.LocationStack do
                        (m.File, m.StartLine, m.StartColumn, m.EndLine, m.EndColumn) |]
            (diag.Severity, "SC", diag.Number, stack, diag.Message) |]

[<AutoOpen>]
module MoreTools =

    type BaseModel with

        /// Analyses the shapes of a model and prints a report
        static member AnalyseShapes<'T when 'T :> DiffSharp.Model.Model> ([<CallerFilePath>] caller, [<CallerLineNumber>] callerLine, ?optionals: bool) =
            let optionals = defaultArg optionals true
            let dflt = Backend.Default
            try
                Backend.Default <- Backend.ShapeChecking
                let syms = SymScope()
                let ctors = typeof<'T>.GetConstructors()
                let ctor = 
                    ctors 
                    |> Array.tryFind (fun ctor -> 
                        ctor.GetParameters() |> Array.exists (fun p -> 
                            let pt = p.ParameterType.ToString()
                            pt.Contains("DiffSharp.Int")))
                    |> function 
                        | None -> ctors.[0] // failwith "couldn't find a model constructor taking Int parameter"
                        | Some c -> c
                let loc = { File = caller; StartLine = callerLine; StartColumn = 0; EndLine = callerLine; EndColumn= 80 }

                // TODO: use _diags
                let ctorArgs, model, methCalls, _diags = makeModelAndRunShapeChecks syms optionals ctor [| |] loc [| |]
                match model with 
                | Error e -> 
                   printfn "%O: error DS1998 - %s" loc e.Message
                | Ok model -> 
                let model = model :?> Model

                printfn ""
                printfn "---------------------"
                let argText = 
                    (ctor.GetParameters(), Array.map snd ctorArgs) 
                    ||> Array.zip
                    |> Array.filter (fun (_p, arg) -> arg <> null) // filter out 'None'
                    |> Array.map (fun (p, arg) -> 
                        // get rid of the Some for F# optional arguments
                        p, (if arg.GetType().FullName.StartsWith("Microsoft.FSharp.Core.FSharpOption`1[") then 
                                (snd (Reflection.FSharpValue.GetUnionFields(arg, arg.GetType()))).[0] 
                            else arg)) // filter out 'None'
                    |> Array.map (fun (p, arg) -> if p.IsOptional then p.Name+"="+string arg else string arg)
                    |> String.concat ","
                printfn "%s(%s)" typeof<'T>.FullName argText
                      
                for (KeyValue(a,b)) in model.parameters.values |> Seq.toArray |> Seq.sortBy (fun (KeyValue(a,_)) -> a) do
                    printfn "   %s : %O" a b.value.shapex

                // Probe the forward function for shape behaviour
                for (m, input, res) in methCalls do
                    match res with 
                    | Ok res -> 
                        printfn "   %s(%O) : %O" m.Name input res
                    | Error e -> 
                        printfn "   %s(%O) // Error: %s" m.Name input e.Message
                        //printfn "      %O --> fails\n%s" input.shapex (e.ToString().Split('\r','\n') |> Array.map (fun s -> "        " + s) |> String.concat "\n")
                        () 
            finally
                Backend.Default <- dflt
