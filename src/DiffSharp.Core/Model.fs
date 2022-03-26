// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp
open DiffSharp.Util
open System.Collections
open System.Collections.Generic
open System.Collections.Specialized


/// <namespacedoc>
///   <summary>Contains types and functionality related to describing models.</summary>
/// </namespacedoc>
///
/// <summary>Represents a parameter.</summary>
/// <remarks>A parameter is a mutable register holding a tensor.</remarks>
type Parameter =
    val mutable value:Tensor
    new(value) = {value=value}

    /// <summary>TBD</summary>
    member p.forwardDiff(derivative:Tensor, ?nestingTag:uint32) = p.value <- p.value.forwardDiff(derivative, ?nestingTag=nestingTag)

    /// <summary>TBD</summary>
    member p.reverseDiff(?nestingTag:uint32) = p.value <- p.value.reverseDiff(?nestingTag=nestingTag)

    /// <summary>TBD</summary>
    member p.noDiff() = p.value <- p.value.noDiff()

    /// <summary>TBD</summary>
    member p.move(?device, ?dtype, ?backend) = p.value <- p.value.move(?device=device, ?dtype=dtype, ?backend=backend)

    member p.copy() = Parameter(p.value.clone())

    /// <summary>TBD</summary>
    override p.ToString() = sprintf "Parameter(shape: %A, value: %A)" (p.value.shape |> List.ofSeq) p.value


/// <summary>Represents a collection of named parameters.</summary>
type ParameterDict() =
    /// <summary>TBD</summary>
    // A generic Dictionary is not good because it does not guarantee an order in which the items are returned. https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.dictionary-2?view=net-5.0
    // A generic SortedDictionary exists but we don't want to sort the parameters by keys and we want to have them in the order they were registered. https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.sorteddictionary-2?view=net-5.0
    // This non-generic OrderedDictionary is used since there is currently no generic OrderedDictionary https://github.com/dotnet/runtime/issues/24826
    member val private parameters = OrderedDictionary() 

    /// <summary>TBD</summary>
    member d.Item
        with get (key:string) = (d.parameters[key] :?> Parameter).value
        and set (key:string) (v:Tensor) = (d.parameters[key] :?> Parameter).value <- v

    interface IEnumerable<string*Parameter> with
        member d.GetEnumerator():IEnumerator<string*Parameter> = 
            let s = d.parameters |> Seq.cast<DictionaryEntry> |> Seq.map (fun v -> v.Key :?> string, v.Value :?> Parameter) in s.GetEnumerator()

    interface System.Collections.IEnumerable with
        member d.GetEnumerator() = (d :> IEnumerable<string*Parameter>).GetEnumerator() :> System.Collections.IEnumerator

    member d.count = d.parameters.Count

    member d.device
        with get() = 
            if d.parameters.Count = 0 then Device.Default // Empty ParameterDict defaults to default device, dtype, backend config
            else let p = d.parameters[0] :?> Parameter in p.value.device

    member d.dtype
        with get() = 
            if d.parameters.Count = 0 then Dtype.Default // Empty ParameterDict defaults to default device, dtype, backend config
            else let p = d.parameters[0] :?> Parameter in p.value.dtype

    member d.backend
        with get() = 
            if d.parameters.Count = 0 then Backend.Default // Empty ParameterDict defaults to default device, dtype, backend config
            else let p = d.parameters[0] :?> Parameter in p.value.backend

    member d.isForwardDiff
        with get() = 
            if d.parameters.Count = 0 then false
            else let p = d.parameters[0] :?> Parameter in p.value.isForwardDiff

    member d.isReverseDiff
        with get() = 
            if d.parameters.Count = 0 then false
            else let p = d.parameters[0] :?> Parameter in p.value.isReverseDiff

    member d.isNoDiff
        with get() = 
            if d.parameters.Count = 0 then true
            else let p = d.parameters[0] :?> Parameter in p.value.isNoDiff

    /// <summary>TBD</summary>
    member d.clear() = d.parameters.Clear()

    /// <summary>TBD</summary>
    member d.add(name, parameter:Parameter) = 
        if d.device <> parameter.value.device then failwithf "Expecting a parameter with device %A but received %A" d.device parameter.value.device
        if d.dtype <> parameter.value.dtype then failwithf "Expecting a parameter with dtype %A but received %A" d.dtype parameter.value.dtype
        if d.backend <> parameter.value.backend then failwithf "Expecting a parameter with backend %A but received %A" d.backend parameter.value.backend
        d.parameters.Add(name, parameter)

    /// <summary>TBD</summary>
    member d.add(parameters:list<string*Parameter>) = for (n, p) in parameters do d.add(n, p)

    /// <summary>TBD</summary>
    member d.add(parameters:ParameterDict) = for n, p in parameters do d.add(n, p)

    /// <summary>TBD</summary>
    member d.map(f:string*Parameter->string*Parameter) =
        let ret = ParameterDict()
        for n, p in d do 
            let n, p = f(n, p)
            ret.add(n, p)
        ret

    /// <summary>TBD</summary>
    member d.map(f:Parameter->Parameter) = d.map(fun (n, p) -> (n, f p))

    /// <summary>TBD</summary>
    member d.copy() = d.map(fun (n, p:Parameter) -> (n, p.copy()))

    /// <summary>TBD</summary>
    member d.set(other:ParameterDict) = 
        let dKeys = d.parameters.Keys
        let oKeys = other.parameters.Keys
        if dKeys <> oKeys then failwithf "Expecting ParameterDict objects to have same set of keys."
        d.iter(fun (n, p) -> p.value <- other[n])

    /// <summary>TBD</summary>
    member d.iter(f:string*Parameter->unit) = for n, p in d do f(n, p)

    /// <summary>
    ///  Adjust the parameters to include support for forward-mode automatic differentiation.
    /// </summary>
    /// <param name="derivatives">The derivatives of the parameters</param>
    /// <param name="nestingTag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters in this dictionary will have attached derivatives for forward mode differentiation.
    /// </remarks>
    member d.forwardDiff(derivatives:ParameterDict, ?nestingTag:uint32) = 
        // This is to be extra cautious about all Parameters in the ParameterDict getting the same tag, which is crucial for correctness of differentiation results
        // If we leave the default tag value to be determined by each underlying tensor, there is a risk that the tag can somehow change during the ParameterDict .iter call
        let nestingTag = defaultArg nestingTag GlobalNestingLevel.Current
        d.iter(fun (n, p) -> p.forwardDiff(derivatives[n], nestingTag=nestingTag))

    /// <summary>
    ///  Adjust the parameters to include support for reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="nestingTag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters in this dictionary will support reverse-mode differentiation. After the completion
    ///  of the corresponding <c>reverse</c> operation, the computed derivative
    ///  will be available. 
    /// </remarks>
    member d.reverseDiff(?nestingTag:uint32) = 
        // This is to be extra cautious about all Parameters in the ParameterDict getting the same tag, which is crucial for correctness of differentiation results
        // If we leave the default tag value to be determined by each underlying tensor, there is a risk that the tag can somehow change during the ParameterDict .iter call
        let nestingTag = defaultArg nestingTag GlobalNestingLevel.Current
        d.iter(fun (_, p) -> p.reverseDiff(nestingTag=nestingTag))

    /// <summary>TBD</summary>
    member d.noDiff() = d.iter(fun (_, p) -> p.noDiff())

    /// <summary>TBD</summary>
    member d.move(?device, ?dtype, ?backend) =
        d.iter (fun (_, p) -> p.move(?device=device, ?dtype=dtype, ?backend=backend))

    /// <summary>TBD</summary>
    member d.nelement with get() = [|for t in d.parameters.Values do (t :?> Parameter).value.nelement|] |> Array.sum

    /// <summary>TBD</summary>
    member d.flatten() =
        if d.count = 0 then dsharp.zeros(0) // Empty ParameterDict defaults to default device, dtype, backend config
        elif d.isReverseDiff then
            // We flatten reverse-mode parameters into a single reverse-mode tensor and also keep the derivative information to cover the use case
            // where the reverse-mode derivative of the parameters (after reverse propagation) can be read from the flattened tensor.
            // This extra code is needed because for reverse-mode tensor operations like cat normally do not keep derivative information and only apply to primals.
            // This mirrors the behavior in ParameterDict.unflatten.
            let pp, pd = Array.unzip [| for t in d.parameters.Values do let t = (t :?> Parameter) in t.value.primal.view(-1), t.value.derivative.view(-1) |]
            let tp, td = dsharp.cat(pp), dsharp.cat(pd)
            tp.reverseDiff(derivative=td, nestingTag=(d.parameters[0] :?> Parameter).value.nestingTag)
        else
        let ts = [| for t in d.parameters.Values do (t :?> Parameter).value.view(-1) |]
        dsharp.cat(ts)

    /// <summary>TBD</summary>
    member d.unflatten(tensors:Tensor) =
        if tensors.dim <> 1 then failwithf "Expecting 1d tensors but received tensors with shape %A" tensors.shape
        if tensors.nelement <> d.nelement then failwithf "Expecting tensors.nelement (%A) and ParameterDict.nelement (%A) to be the same" tensors.nelement d.nelement
        if tensors.nelement = 0 then ()
        else
        let shapes = [|for t in d.parameters.Values do (t :?> Parameter).value.shape|]
        let sizes = [|for s in shapes do shapeLength s|]
        let ts = 
            if tensors.isReverseDiff && tensors.derivative.shape = tensors.primal.shape then
                // For reverse-mode tensors, we split both primals and derivatives and combine these in reverse-mode tensors corresponding to each parameter
                // This extra code is needed because reverser-mode tensor operations like split normally do not keep derivative information and only apply to primals.
                // This mirrors the behavior in ParameterDict.flatten.
                Array.map3 (fun (tp:Tensor) (td:Tensor) (s:int[]) -> tp.view(s).reverseDiff(derivative=td.view(s), nestingTag=tensors.nestingTag)) (tensors.primal.split(sizes)) (tensors.derivative.split(sizes)) shapes
            else
                Array.map2 (fun (t:Tensor) (s:int[]) -> t.view(s)) (tensors.split(sizes)) shapes
        let mutable i = 0
        let keys = OrderedDictionary.copyKeys d.parameters
        for n in keys do
            d[n] <- ts[i]
            i <- i+1

    /// <summary>TBD</summary>
    member d.unflattenToNew(tensors:Tensor) = 
        let dd = d.copy()
        dd.unflatten(tensors)
        dd

    /// <summary>TBD</summary>
    override d.ToString() =
        if d.parameters.Count = 0 then "ParameterDict()"
        else
        let sb = System.Text.StringBuilder()
        sb.AppendLine("ParameterDict(") |> ignore
        for n, p in d do 
            sb.AppendLine(sprintf "%A: %A" n p) |> ignore
        sb.Append(")") |> ignore
        sb.ToString()


/// <summary>Indicates the training or evaluation mode for a model.</summary>
type Mode =
    | Train = 0
    | Eval = 1


/// <summary>Represents the base class of all models.</summary>
[<AbstractClass>]
type ModelBase() =
    [<DefaultValue>]
    val mutable mode: Mode

    /// <summary>TBD</summary>
    let namePrefixes = Dictionary<string, int>()
    let parameterDict = ParameterDict()
    let bufferDict = ParameterDict()
    let stateDict = ParameterDict()
    let modelDict = OrderedDictionary()

    let updateState() =
        stateDict.clear()
        stateDict.add(parameterDict)
        stateDict.add(bufferDict)

    let nextName (name:string) =
        let name = if name.Contains("__") then name.Split("__")[0] else name
        let i = namePrefixes.GetValueOrDefault name
        namePrefixes[name] <- i+1
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
        for model:ModelBase in m.descendants do model.mode <- Mode.Train

    /// <summary>TBD</summary>
    member m.eval() = 
        m.mode <- Mode.Eval
        for model:ModelBase in m.descendants do model.mode <- Mode.Eval

    member _.device
        with get() = parameterDict.device

    member _.dtype
        with get() = parameterDict.dtype

    member _.backend
        with get() = parameterDict.backend

    member _.isForwardDiff
        with get() = parameterDict.isForwardDiff

    member _.isReverseDiff
        with get() = parameterDict.isReverseDiff

    member _.isNoDiff
        with get() = parameterDict.isNoDiff

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

    /// <summary>TBD</summary>
    member _.state
        with get () = stateDict
        and set s = stateDict.set(s)

    /// <summary>TBD</summary>
    member _.stateVector
        with get () = stateDict.flatten()
        and set s = stateDict.unflatten(s)

    /// <summary>Gets the number of parameters of the Model</summary>
    member m.nparameters = m.parameters.nelement

    /// <summary>TBD</summary>
    member m.nbuffers = m.buffers.nelement

    /// <summary>TBD</summary>
    member m.nstate = m.state.nelement

    /// <summary>TBD</summary>
    member _.children
        with get () = 
            modelDict.Values |> Seq.cast<ModelBase> |> Seq.toList

    /// <summary>TBD</summary>
    member m.descendants
        with get () =
            m :: [for child in m.children do yield! child.descendants]

    /// <summary>TBD</summary>
    member m.hasOwnParameters
        with get () =
            let childrenParams = m.children |> List.map (fun c -> c.nparameters) |> List.sum
            m.nparameters <> childrenParams

    /// <summary>TBD</summary>
    member m.hasOwnBuffers
        with get () =
            let childrenBuffers = m.children |> List.map (fun c -> c.nbuffers) |> List.sum
            m.nbuffers <> childrenBuffers

    /// <summary>TBD</summary>
    member m.hasOwnState
        with get () =
            let childrenState = m.children |> List.map (fun c -> c.nstate) |> List.sum
            m.nstate <> childrenState

    /// <summary>TBD</summary>
    member m.init(f:string*Tensor->Tensor) = m.parameters.iter(fun (n, p) -> p.value <- f(n, p.value))

    /// <summary>TBD</summary>
    member m.addParameter(items:seq<Parameter>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let param = items[i]
            let n = if names.Length > 0 then names[i] else sprintf "Parameter-%s" (Random.UUID())
            parameterDict.add(n, param)
        updateState()

    /// <summary>TBD</summary>
    member m.addBuffer(items:seq<Parameter>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let param = items[i]
            let n = if names.Length > 0 then names[i] else sprintf "Buffer-%s" (Random.UUID())
            bufferDict.add(n, param)
        updateState()

    /// <summary>TBD</summary>
    member m.addModel(items:seq<obj>, ?names:seq<string>) =
        let items, names = m.checkItems(items, ?names=names)
        for i in 0..items.Length-1 do
            let model = 
                match items[i] with
                | :? ModelBase as mm -> mm
                | _ -> failwithf "Unsupported type. Expecting a ModelBase."
            let n = if names.Length > 0 then names[i] else sprintf "Model-%s" (Random.UUID())

            modelDict.Add(n, model)
            for n, p in model.parameters do 
                parameterDict.add(nextName n, p)
            for n, b in model.buffers do 
                bufferDict.add(nextName n, b)
        updateState()

    /// <summary>
    ///  Adjust the parameters of the model to initiate a new level of forward-mode automatic differentiation.
    /// </summary>
    /// <param name="derivatives">The derivatives of the parameters</param>
    /// <param name="nestingTag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will have attached derivatives for forward mode differentiation.
    /// </remarks>
    member m.forwardDiff(derivatives:ParameterDict, ?nestingTag) = m.parameters.forwardDiff(derivatives, ?nestingTag=nestingTag)

    /// <summary>
    ///  Adjust the parameters of the model to initiate a new level of reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="nestingTag">The level tag for nested differentiation.  Defaults to the current global nesting level</param>
    /// <remarks>
    ///  After this call the current parameters of the model will support reverse-mode differentiation. After the completion
    ///  of the corresponding <c>reverse</c> operation, the computed derivatives will be available. 
    /// </remarks>
    member m.reverseDiff(?nestingTag) = m.parameters.reverseDiff(?nestingTag=nestingTag)

    /// <summary>TBD</summary>
    member m.noDiff() = m.parameters.noDiff()

    /// <summary>Moves the state (parameters and buffers) of the model to the given configuration</summary>
    member m.move(?device, ?dtype, ?backend) = 
        m.state.move(?device=device, ?dtype=dtype, ?backend=backend)

    /// <summary>TBD</summary>
    member m.saveState(fileName, ?noDiff:bool) =
        let noDiff = defaultArg noDiff true
        let ss =
            if noDiff then m.stateVector.noDiff() // We remove any derivatives from the state vector before saving. This doesn't alter the differentiation state of the model.
            else m.stateVector
        ss.save(fileName)

    /// <summary>TBD</summary>
    member m.loadState(fileName) = m.stateVector <- Tensor.load(fileName)

    /// <summary>TBD</summary>
    member m.save(fileName, ?noDiff:bool) =
        let noDiff = defaultArg noDiff true
        let mm =
            if noDiff then
                if m.isNoDiff then m
                else
                    // We clone the model and then remove any derivatives before saving. 
                    // The clone is used because we don't want a save operation to alter the differentiation state of the model.
                    let mClone:ModelBase = m.clone()
                    mClone.noDiff()
                    mClone
            else m
        saveBinary mm fileName

    /// <summary>TBD</summary>
    static member load(fileName):ModelBase = loadBinary fileName

    /// <summary>TBD</summary>
    member m.clone() = 
        let fileName = System.IO.Path.GetTempFileName()
        m.save(fileName, noDiff=false)
        ModelBase.load(fileName)

    override m.ToString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("Model(") |> ignore
        let mutable prefix = ""
        for mm in m.descendants do
            if mm.hasOwnParameters then
                sb.Append(sprintf "%s%A" prefix mm) |> ignore
                prefix <- ", "
        sb.Append(")") |> ignore
        sb.ToString()

    /// <summary>TBD</summary>
    member m.summary() =
        let sb = System.Text.StringBuilder()
        sb.AppendLine("---") |> ignore
        sb.AppendLine(sprintf "%-40s %16s" "Model" "Params") |> ignore
        sb.AppendLine("---") |> ignore
        for mm in m.descendants do
            if mm.hasOwnParameters then
                sb.AppendLine(sprintf "%-40s %16s" (mm.ToString()) (thousandsInt mm.nparameters)) |> ignore
        sb.AppendLine("---") |> ignore
        sb.AppendLine(sprintf "Total params                  : %s" (thousandsInt m.nstate)) |> ignore
        sb.AppendLine(sprintf "Trainable params              : %s" (thousandsInt m.nparameters)) |> ignore
        sb.AppendLine(sprintf "Non-trainable params (buffers): %s" (thousandsInt m.nbuffers)) |> ignore
        sb.AppendLine("---") |> ignore
        sb.AppendLine(sprintf "Total params size (MiB)       : %s" (thousandsFloat ((float m.stateVector.memorySize)/(1024.*1024.)))) |> ignore
        sb.AppendLine("---") |> ignore
        sb.ToString()



/// <summary>Represents a model, primarily a collection of named parameters and sub-models and a function governed by them.</summary>
[<AbstractClass>]
type Model<'In, 'Out>() =
    inherit ModelBase()

    /// <summary>TBD</summary>
    abstract member forward: 'In -> 'Out

    /// <summary>Use the model as a function of its parameters and input.</summary>
    /// <remarks>
    ///    The resulting function can be composed with a loss function and differentiated.
    ///    During execution the parameters of the model are temporarily set to the supplied parameters.
    /// </remarks>
    member m.asFunction (parameters:Tensor) (input:'In) =
        let old = m.parametersVector
        try 
            m.parametersVector <- parameters
            m.forward(input)
        finally
            m.parametersVector <- old
    
    /// <summary>TBD</summary>
    static member create (models: seq<obj>) (parameters: seq<Parameter>) (buffers: seq<Parameter>) (f: 'In -> 'Out) : Model<'In, 'Out> =
        let model = { new Model<'In, 'Out>() with override _.forward(x:'In) : 'Out = f x}
        model.addModel(models)
        model.addParameter(parameters)
        model.addBuffer(buffers)
        model

    /// <summary>TBD</summary>
    static member compose (model1:Model<'In, 'Out>) (model2:Model<'Out, 'Out2>) : Model<'In, 'Out2> =
        Model<'In, 'Out2>.create [box model1; box model2] [] [] (model1.forward >> model2.forward)

    /// <summary>TBD</summary>
    static member (-->) (model1:Model<'In, 'Out>, model2:Model<'Out, 'Out2>) = Model<'In, 'Out>.compose model1 model2
    
    /// <summary>TBD</summary>
    static member (-->) (model:Model<'In, 'Out>, f:'Out->'Out2) = Model<'In, 'Out2>.create [model] [] [] (model.forward >> f)

    /// <summary>TBD</summary>
    static member (-->) (f:'In->'Out, model:Model<'Out, 'Out2>) = Model<'In, 'Out2>.create [model] [] [] (f >> model.forward)

    /// <summary>TBD</summary>
    static member (-->) (t:'In, model:Model<'In, 'Out>) = model.forward t

    /// <summary>TBD</summary>
    static member load(fileName):Model<'In, 'Out> = ModelBase.load(fileName) :?> Model<'In, 'Out>

    /// <summary>TBD</summary>
    member m.clone():Model<'In, 'Out> = (m :> ModelBase).clone() :?> Model<'In, 'Out>


type Model = Model<Tensor, Tensor>


/// <summary>Contains functionality related to generating initial parameter weights for models.</summary>
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
