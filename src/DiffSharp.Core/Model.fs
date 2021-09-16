// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Model

open DiffSharp
open DiffSharp.Util
open System.Collections.Generic


/// <namespacedoc>
///   <summary>Contains types and functionality related to describing models.</summary>
/// </namespacedoc>
///
/// <summary>Represents a parameter in a model.</summary>
/// <remarks>A parameter is a mutable register holding a tensor.</remarks>
type Parameter =
    val mutable value:Tensor
    new(value) = {value=value}

    /// <summary>TBD</summary>
    member p.forwardDiff(derivative:Tensor, ?tag:uint32) = p.value <- p.value.forwardDiff(derivative, ?tag=tag)

    /// <summary>TBD</summary>
    member p.reverseDiff(?tag:uint32) = p.value <- p.value.reverseDiff(?tag=tag)

    /// <summary>TBD</summary>
    member p.noDiff() = p.value <- p.value.noDiff()

    /// <summary>TBD</summary>
    member p.move(?device, ?dtype, ?backend) = p.value <- p.value.move(?device=device, ?dtype=dtype, ?backend=backend)

    /// <summary>TBD</summary>
    override p.ToString() = sprintf "Parameter(shape:%A, value:%A)" p.value.shape p.value


/// <summary>Represents a collection of named parameters in a model.</summary>
type ParameterDict() =

    // If the dictionary is empty then the latest 'move' is considered the configuration for the implied empty tensor
    let mutable dummy = dsharp.zeros(0)

    /// <summary>TBD</summary>
    member val values = Dictionary<string, Parameter>()

    /// <summary>TBD</summary>
    member d.Item
        with get key = d.values.[key].value
        and set key v = d.values.[key].value <- v

    /// <summary>TBD</summary>
    member d.add(name, parameter) = d.values.Add(name, parameter)

    /// <summary>TBD</summary>
    member d.add(parameters:list<string*Parameter>) = for (n, p) in parameters do d.add(n, p)

    /// <summary>TBD</summary>
    member d.add(parameters:ParameterDict) = for KeyValue(n, p) in parameters.values do d.add(n, p)

    /// <summary>TBD</summary>
    member d.copy() = d.map(fun (t:Tensor) -> t)

    /// <summary>TBD</summary>
    member d.map(f:string*Parameter->string*Parameter) =
        let ret = ParameterDict()
        for KeyValue(n, p) in d.values do ret.values.Add(f(n,p))
        ret

    /// <summary>TBD</summary>
    member d.map(f:string*Tensor->string*Tensor) = d.map(fun (n, p:Parameter) -> let nn, tt = f(n, p.value) in nn, Parameter(tt))

    /// <summary>TBD</summary>
    member d.map(f:Tensor->Tensor) = d.map(fun (n,t) -> n, f t)

    /// <summary>TBD</summary>
    member d.set(parameters:ParameterDict) = d.iter(fun (n, p) -> p.value <- parameters.[n])

    /// <summary>TBD</summary>
    member d.iter(f:string*Parameter->unit) = for KeyValue(n, p) in d.values do f(n,p)

    /// <summary>
    ///  Adjust the parameters to include support for forward-mode automatic differentiation.
    /// </summary>
    /// <param name="derivatives">The derivatives of the parameters</param>
    /// <param name="tag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will have attached derivatives for forward mode propagation.
    /// </remarks>
    member d.forwardDiff(derivatives:ParameterDict, ?tag:uint32) = 
        // This is to be extra cautious about all Parameters in the ParameterDict getting the same tag, which is crucial for correctness of differentiation results
        // If we leave the default tag value to be determined by each underlying tensor, there is a risk that the tag can somehow change during the ParameterDict .iter call
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (n, p) -> p.forwardDiff(derivatives.[n], tag=tag))

    /// <summary>
    ///  Adjust the parameters to include support for reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="tag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will support reverse-mode propagation. After the completion
    ///  of the corresponding <c>reverse</c> operation, the computed derivative
    ///  will be available. 
    /// </remarks>
    member d.reverseDiff(?tag:uint32) = 
        // This is to be extra cautious about all Parameters in the ParameterDict getting the same tag, which is crucial for correctness of differentiation results
        // If we leave the default tag value to be determined by each underlying tensor, there is a risk that the tag can somehow change during the ParameterDict .iter call
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (_, p) -> p.reverseDiff(tag=tag))

    /// <summary>TBD</summary>
    member d.noDiff() = d.iter(fun (_, p) -> p.noDiff())

    /// <summary>TBD</summary>
    member d.move(?device, ?dtype, ?backend) =
        dummy <- dummy.move(?device=device, ?dtype=dtype, ?backend=backend)
        d.iter (fun (_, p) -> p.move(?device=device, ?dtype=dtype, ?backend=backend))

    /// <summary>TBD</summary>
    member d.primal with get() = d.map(fun (t:Tensor)->t.primal)

    /// <summary>TBD</summary>
    member d.derivative with get() = d.map(fun (t:Tensor)->t.derivative)

    /// <summary>TBD</summary>
    member d.nelement with get() = [|for t in d.values.Values do t.value.nelement|] |> Array.sum

    /// <summary>TBD</summary>
    member d.flatten() =
        let ts = [| for t in d.values.Values do t.value.view(-1) |]
        if ts.Length = 0 then dummy else
        dsharp.cat(ts)

    /// <summary>TBD</summary>
    member d.unflatten(tensors:Tensor) =
        if tensors.dim <> 1 then failwithf "Expecting 1d tensors but received tensors with shape %A" tensors.shape
        if tensors.nelement <> d.nelement then failwithf "Expecting tensors.nelement (%A) and ParameterDict.nelement (%A) to be the same" tensors.nelement d.nelement
        let shapes = [|for t in d.values.Values do t.value.shape|]
        let sizes = [|for s in shapes do shapeLength s|]
        let ts = Array.map2 (fun (t:Tensor) (s:int[]) -> t.view(s)) (tensors.split(sizes)) shapes
        let mutable i = 0
        let keys = Dictionary.copyKeys d.values
        for n in keys do
            d.[n] <- ts.[i]
            i <- i+1

    /// <summary>TBD</summary>
    member d.unflattenToNew(tensors:Tensor) = 
        let dd = d.copy()
        dd.unflatten(tensors)
        dd

    /// <summary>TBD</summary>
    override d.ToString() =
        let sb = System.Text.StringBuilder()
        sb.Append("ParameterDict(") |> ignore
        let mutable prefix = ""
        for KeyValue(n, p) in d.values do 
            sb.Append(sprintf "%s%A:%A" prefix n p) |> ignore
            prefix <- ", "
        sb.Append(")") |> ignore
        sb.ToString()


/// <summary>Indicates the training or evaluation mode for a model.</summary>
type Mode =
    | Train = 0
    | Eval = 1


/// <summary>Represents a model, primarily a collection of named parameters and sub-models and a function governed by them.</summary>
[<AbstractClass>]
type ModelBase() =
    [<DefaultValue>]
    val mutable mode: Mode

    /// <summary>TBD</summary>
    let namePrefixes = Dictionary<string, int>()
    let parameterDict = ParameterDict()
    let bufferDict = ParameterDict()
    let mutable stateDict = ParameterDict()
    let modelDict = Dictionary<string, ModelBase>()

    let updateState() =
        stateDict <- parameterDict.copy()
        stateDict.add(bufferDict)

    let nextName (name:string) =
        let name = if name.Contains("__") then name.Split("__").[0] else name
        let i = namePrefixes.GetValueOrDefault name
        namePrefixes.[name] <- i+1
        sprintf "%s__%A" name (i+1)

    member _.checkItems(items:seq<_>, ?names:seq<string>)=
        let items = items |> Seq.toArray
        let names = defaultArg names (Seq.empty) |> Seq.toArray
        if names.Length > 0 then
            if items.Length <> names.Length then failwithf "Expecting items (%A) and names (%A) to have the same length" items.Length names.Length
            for name in names do if name.Contains("__") then failwithf "String '__' not allowed in name '%s'" name
        items, names

    /// <summary>TBD</summary>
    member m.train() = 
        m.mode <- Mode.Train
        for model:ModelBase in m.models do model.mode <- Mode.Train

    /// <summary>TBD</summary>
    member m.eval() = 
        m.mode <- Mode.Eval
        for model:ModelBase in m.models do model.mode <- Mode.Eval

    /// <summary>TBD</summary>
    member _.parameters
        with get () = parameterDict
        and set p = parameterDict.set(p)

    /// <summary>TBD</summary>
    member _.parametersVector
        with get () = parameterDict.flatten()
        and set p = parameterDict.unflatten(p)

    /// <summary>TBD</summary>
    member _.buffers
        with get () = bufferDict
        and set b = bufferDict.set(b)

    /// <summary>TBD</summary>
    member _.buffersVector
        with get () = bufferDict.flatten()
        and set b = bufferDict.unflatten(b)

    member _.state
        with get () = stateDict
        and set s = stateDict.set(s)

    member _.stateVector
        with get () = stateDict.flatten()
        and set s = stateDict.unflatten(s)

    member _.children
        with get () = 
            modelDict.Values |> Seq.toList

    member m.models
        with get () =
            m :: [for c in m.children do yield! c.models] 

    /// <summary>TBD</summary>
    member m.init(f:string*Tensor->Tensor) = for KeyValue(n, p) in m.parameters.values do p.value <- f(n, p.value)

    member m.addParameter(items:seq<Parameter>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let param = items.[i]
            let n = if names.Length > 0 then names.[i] else sprintf "param-%s" (Random.UUID())
            parameterDict.add(nextName n, param)
        updateState()

    member m.addBuffer(items:seq<Parameter>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let param = items.[i]
            let n = if names.Length > 0 then names.[i] else sprintf "buffer-%s" (Random.UUID())
            bufferDict.add(nextName n, param)
        updateState()

    member m.addModel(items:seq<Model>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let model = items.[i]
            let n = if names.Length > 0 then names.[i] else sprintf "model-%s" (Random.UUID())
            modelDict.Add(n, model)
            parameterDict.add(model.parameters.map(fun (nn, pp:Parameter) -> (nextName nn, pp)))
        updateState()

    member m.addParameter(item:Parameter, ?name:string) = m.addParameter([item], ?names=if name.IsSome then Some(seq {name.Value}) else None)
    member m.addBuffer(item:Parameter, ?name:string) = m.addBuffer([item], ?names=if name.IsSome then Some(seq {name.Value}) else None)
    member m.addModel(item:Model, ?name:string) = m.addModel([item], ?names=if name.IsSome then Some(seq {name.Value}) else None)

    /// <summary>TBD</summary>
    member _.add(items:seq<obj>, ?names:seq<string>) =
        let items = items |> Seq.toArray
        let names = defaultArg names (Seq.empty) |> Seq.toArray
        if names.Length > 0 then
            if items.Length <> names.Length then failwithf "Expecting items (%A) and names (%A) to have the same length" items.Length names.Length
            for name in names do if name.Contains("__") then failwithf "String '__' not allowed in name '%s'" name
        for i in 0..items.Length-1 do
            let item = items.[i]
            match item with
            | :? Parameter as param ->
                let n = if names.Length > 0 then names.[i] else sprintf "param-%s" (Random.UUID())
                parameterDict.add(n, param)
            | :? Model as model ->
                let n = if names.Length > 0 then names.[i] else sprintf "model-%s" (Random.UUID())
                modelDict.Add(n, model)
                parameterDict.add(model.parameters.map(fun (nn, pp:Parameter) -> (nextName nn, pp)))
            | _ -> failwithf "Unsupported type. Expecting a Parameter or Model"

    /// <summary>
    ///  Adjust the parameters of the model to include support for forward-mode automatic differentiation.
    /// </summary>
    /// <param name="derivatives">The derivatives of the parameters</param>
    /// <param name="tag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will have attached derivatives for forward mode propagation.
    /// </remarks>
    member m.forwardDiff(derivatives:ParameterDict, ?tag) = m.parameters.forwardDiff(derivatives, ?tag=tag)

    /// <summary>
    ///  Adjust the parameters of the model to include support for reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="tag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will support reverse-mode propagation. After the completion
    ///  of the corresponding <c>reverse</c> operation, the computed derivative will be available. 
    /// </remarks>
    member m.reverseDiff(?tag) = m.parameters.reverseDiff(?tag=tag)

    /// <summary>TBD</summary>
    member m.noDiff() = m.parameters.noDiff()

    /// <summary>Moves the parameters of the model to the given configuration</summary>
    member m.move(?device, ?dtype, ?backend) = m.parameters.move(?device=device, ?dtype=dtype, ?backend=backend)

    /// <summary>Gets the number of parameters of the model</summary>
    member m.nparameters = m.parameters.nelement

    /// <summary>TBD</summary>
    member m.saveParameters(fileName) = m.parametersVector.save(fileName)

    /// <summary>TBD</summary>
    member m.loadParameters(fileName) = m.parametersVector <- Tensor.load(fileName)

    /// <summary>TBD</summary>
    member m.save(fileName) = saveBinary m fileName


[<AbstractClass>]
type Model<'In, 'Out>() =
    inherit ModelBase()

    /// <summary>TBD</summary>
    abstract member forward: 'In -> 'Out

    /// <summary>Use the model as a function of its input and parameters</summary>
    /// <remarks>
    ///    The resulting function can be composed with a loss function and differentiated.
    ///    During execution the parameters of the model are temporarily set to the supplied parameters.
    /// </remarks>
    member m.asFunction (input:'In) (parameters:Tensor) =
        let old = m.parametersVector
        try 
            m.parametersVector <- parameters
            m.forward(input) 
        finally
            m.parametersVector <- old

    /// <summary>TBD</summary>
    static member create (ps: seq<obj>) (f: 'In -> 'Out) : Model<'In, 'Out> =
        let model = { new Model<'In, 'Out>() with override _.forward(x:'In) : 'Out = f x}
        model.add(ps)
        model

    /// <summary>TBD</summary>
    static member compose (m1:Model<'In, 'Out>) (m2:Model<'Out, 'Out2>) : Model<'In, 'Out2> =
        Model<'In, 'Out2>.create [box m1; box m2] (m1.forward >> m2.forward)

    /// <summary>TBD</summary>
    static member (-->) (m1:Model<'In, 'Out>, m2:Model<'Out, 'Out2>) = Model<'In, 'Out>.compose m1 m2
    
    /// <summary>TBD</summary>
    static member (-->) (m:Model<'In, 'Out>, f:'Out->'Out2) = Model<'In, 'Out2>.create [m] (m.forward >> f)

    /// <summary>TBD</summary>
    static member (-->) (f:'In->'Out, m:Model<'Out, 'Out2>) = Model<'In, 'Out2>.create [m] (f >> m.forward)

    /// <summary>TBD</summary>
    static member (-->) (t:'In, m:Model<'In, 'Out>) = m.forward t

    /// <summary>TBD</summary>
    static member load(fileName):Model<'In, 'Out> = loadBinary fileName

    /// <summary>TBD</summary>
    member m.clone():Model<'In, 'Out> = 
        let fileName = System.IO.Path.GetTempFileName()
        m.save(fileName)
        Model<'In, 'Out>.load(fileName)


type Model = Model<Tensor, Tensor>


/// <summary>Contains functionality related to generating initial parameter weights.</summary>
type Weight =

    /// <summary>TBD</summary>
    static member kaiming(fanIn, fanOut, ?a:float) = 
        // He et al. 2015. https://arxiv.org/abs/1502.01852
        let a = defaultArg a (sqrt 5.)
        let w = dsharp.randn([fanIn; fanOut])
        let s = sqrt (2. / ((1. + a*a) * (float fanIn)))
        w * s

    /// <summary>TBD</summary>
    static member uniform(shape:Shape, k:float) =
        -k + dsharp.rand(shape) * 2*k
