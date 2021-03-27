// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Model

open DiffSharp
open DiffSharp.Util
open DiffSharp.ShapeChecking
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
    member p.move(?dtype, ?device, ?backend) = p.value <- p.value.move(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    override p.ToString() = sprintf "Parameter(%A)" p.value


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

    /// <summary>TBD</summary>
    member d.forwardDiff(derivatives:ParameterDict, ?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (n, p) -> p.forwardDiff(derivatives.[n], tag))

    /// <summary>TBD</summary>
    member d.reverseDiff(?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (_, p) -> p.reverseDiff(tag))

    /// <summary>TBD</summary>
    member d.noDiff() = d.iter(fun (_, p) -> p.noDiff())

    /// <summary>TBD</summary>
    member d.move(?dtype, ?device, ?backend) = 
        dummy <- dummy.move(?dtype=dtype, ?device=device, ?backend=backend)
        d.iter (fun (_, p) -> p.move(?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    member d.primal with get() = d.map(fun (t:Tensor)->t.primal)

    /// <summary>TBD</summary>
    member d.derivative with get() = d.map(fun (t:Tensor)->t.derivative)

    /// <summary>TBD</summary>
    member d.nelement with get() = [|for t in d.values.Values do t.value.nelement|] |> Array.sum

    /// <summary>TBD</summary>
    member d.nelementx with get() = [|for t in d.values.Values do t.value.nelementx|] |> Array.sum

    /// <summary>TBD</summary>
    member d.flatten() =
        let ts = [| for t in d.values.Values do t.value.view(-1) |]
        if ts.Length = 0 then dummy else
        dsharp.cat(ts)

    /// <summary>TBD</summary>
    member d.unflatten(tensors:Tensor) =
        if tensors.dim <> 1 then failwithf "Expecting 1d tensors but received tensors with shape %A" tensors.shapex
        if not (tensors.nelementx =~= d.nelementx) then failwithf "Expecting tensors.nelementx (%A) and ParameterDict.nelementx (%A) to be the same" tensors.nelementx d.nelementx
        let shapes = [|for t in d.values.Values do t.value.shapex|]
        let sizes = [|for s in shapes do Shape.nelementx s|]
        let ts = Array.map2 (fun (t:Tensor) (s:Shape) -> t.view(s)) (tensors.split(sizes)) shapes
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
type BaseModel() =
    [<DefaultValue>]
    val mutable mode: Mode

    /// <summary>TBD</summary>
    let parameterDict = ParameterDict()
    let parameterPrefixes = Dictionary<string, int>()

    /// <summary>TBD</summary>
    member val SubModelsDict = Dictionary<string, BaseModel>()

    /// <summary>TBD</summary>
    member m.train() = 
        m.mode <- Mode.Train
        for model:BaseModel in m.allModels do model.mode <- Mode.Train

    /// <summary>TBD</summary>
    member m.eval() = 
        m.mode <- Mode.Eval
        for model:BaseModel in m.allModels do model.mode <- Mode.Eval

    /// <summary>TBD</summary>
    member m.parameters
        with get () = parameterDict
        and set parameters = parameterDict.set(parameters)

    /// <summary>TBD</summary>
    member m.parametersVector
        with get () = m.parameters.flatten()
        and set parameters = m.parameters.unflatten(parameters)

    /// <summary>TBD</summary>
    member m.allModels =
        if m.SubModelsDict.Count = 0 then [m]
        else m.subModels

    /// <summary>TBD</summary>
    member m.subModels = [for sm in m.SubModelsDict.Values do yield! sm.allModels]

    /// <summary>TBD</summary>
    member m.init(f:string*Tensor->Tensor) = for KeyValue(n, p) in m.parameters.values do p.value <- f(n, p.value)

    /// <summary>TBD</summary>
    member m.add(parameters:seq<obj>, ?names:seq<string>) =
        let parameters = parameters |> Seq.toArray
        let names = defaultArg names (Seq.empty) |> Seq.toArray
        if names.Length > 0 then
            if parameters.Length <> names.Length then failwithf "Expecting parameters (%A) and names (%A) to have the same length" parameters.Length names.Length
            for name in names do if name.Contains("__") then failwithf "String '__' not allowed in name '%s'" name
        let nextName (name:string) =
            let name = if name.Contains("__") then name.Split("__").[0] else name
            let i = parameterPrefixes.GetValueOrDefault name
            parameterPrefixes.[name] <- i+1
            sprintf "%s__%A" name (i+1)
        let (|Pair|_|) (x: obj) =
            if Reflection.FSharpType.IsTuple (x.GetType()) then  
                match Reflection.FSharpValue.GetTupleFields(x) with
                | [| t1; t2 |] -> Some (t1,t2)
                | _ -> None
            else
                None
        for i in 0..parameters.Length-1 do
            let p = parameters.[i]
            match (box p) with
            | :? Parameter as p ->
                let n = if names.Length > 0 then names.[i] else sprintf "param-%s" (Random.UUID())
                m.parameters.add(n, p)
            | :? Model as mm ->
                let n = if names.Length > 0 then names.[i] else sprintf "model-%s" (Random.UUID())
                m.SubModelsDict.Add(n, mm)
                parameterDict.add(mm.parameters.map(fun (nn, pp:Parameter) -> (nextName nn, pp)))
            | Pair ((:? Parameter as p), (:? string as n))  -> 
                parameterDict.add(n, p)
            | Pair ((:? BaseModel as mm), (:? string as n))  -> 
                m.SubModelsDict.Add(n, mm)
                parameterDict.add(parameterDict.map(fun (nn, pp:Parameter) -> (n + "__" + nn, pp)))
            | t -> failwithf "Unsupported type %A. Expecting a Parameter or Model" (t.GetType())

    /// <summary>TBD</summary>
    member m.forwardDiff(derivatives:ParameterDict) = m.parameters.forwardDiff(derivatives)

    /// <summary>TBD</summary>
    member m.reverseDiff() = m.parameters.reverseDiff()

    /// <summary>TBD</summary>
    member m.noDiff() = m.parameters.noDiff()

    /// <summary>TBD</summary>
    member m.move(?dtype, ?device, ?backend) =
        m.parameters.move(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    member m.nparametersx = m.parameters.nelementx

    /// <summary>TBD</summary>
    member m.nparameters = m.parameters.nelement

    /// <summary>TBD</summary>
    member m.saveParameters(fileName) = m.parametersVector.save(fileName)

    abstract member getString: unit -> string
    default m.getString() =
        if m.allModels |> List.length < 2 then "Model()" // allModels has one element (m) in case there are no submodels
        else
        let sb = System.Text.StringBuilder()
        sb.Append("Model(\n") |> ignore
        for model in m.subModels do sb.Append(sprintf "%A\n" model) |> ignore
        sb.Append(")") |> ignore
        sb.ToString()

    /// <summary>TBD</summary>
    member m.save(fileName) = saveBinary m fileName

    /// <summary>TBD</summary>
    override m.ToString() = sprintf "%s--nparameters:%A" (m.getString()) m.nparametersx

[<AbstractClass>]
type Model<'In, 'Out>() =
    inherit BaseModel()

    /// <summary>TBD</summary>
    abstract member forward: 'In -> 'Out

    /// <summary>TBD</summary>
    static member create (ps: seq<obj>) (f: 'In -> 'Out) : Model<'In, 'Out> =
        let model = { new Model<'In, 'Out>() with override _.forward(x:'In) : 'Out = f x}
        model.add(ps)
        model

    /// <summary>TBD</summary>
    static member compose (m1:Model<'In, 'Out>) (m2:Model<'Out, 'Out2>) : Model<'In, 'Out2> =
        Model<'In, 'Out2>.create [box m1; box m2] (m1.forward >> m2.forward)

    member m.forwardParameters (input:'In) (parameters:Tensor) =
        m.parametersVector <- parameters
        let f = m.forward(input) in m.noDiff(); f

    /// <summary>TBD</summary>
    member m.forwardCompose (f:'Out->'Out2) (input:'In) (parameters:Tensor) =
        m.forwardParameters input parameters |> f

    /// <summary>TBD</summary>
    member m.forwardLoss (f:'In2->'Out->Tensor) (input:'In) (target:'In2) (parameters:Tensor) =
        m.forwardCompose (f target) input parameters

    /// <summary>TBD</summary>
    static member (-->) (m1:Model<'In, 'Out>, m2:Model<'Out, 'Out2>) = Model<'In, 'Out>.compose m1 m2
    
    /// <summary>TBD</summary>
    static member (-->) (m:Model<'In, 'Out>, f:'Out->'Out2) = Model<'In, 'Out2>.create [m] (m.forward >> f)

    /// <summary>TBD</summary>
    static member (-->) (f:'In->'Out, m:Model<'Out, 'Out2>) = Model<'In, 'Out2>.create [m] (f >> m.forward)

    member m.saveParameters(fileName) = m.parametersVector.save(fileName)

    /// <summary>TBD</summary>
    member m.loadParameters(fileName) = m.parametersVector <- Tensor.load(fileName)

    /// <summary>TBD</summary>
    static member (-->) (t:'In, m:Model<'In, 'Out>) = m.forward t

    /// <summary>TBD</summary>
    static member load(fileName):Model<'In, 'Out> = loadBinary fileName

    /// <summary>TBD</summary>
    member m.clone() = 
        let fileName = System.IO.Path.GetTempFileName()
        m.save(fileName)
        Model.load(fileName)

type Model = Model<Tensor, Tensor>

/// <summary>Contains functionality related to generating initial paramerter weights.</summary>
type Weight =

    /// <summary>TBD</summary>
    static member kaiming(fanIn:Int, fanOut:Int, ?a:float) = 
        // He et al. 2015. https://arxiv.org/abs/1502.01852
        let a = defaultArg a (sqrt 5.)
        let w = dsharp.randn([fanIn; fanOut])
        let s = sqrt (2. / ((1. + a*a) * (float fanIn.ValueOrOne)))
        w * s

    /// <summary>TBD</summary>
    static member kaiming(fanIn:int, fanOut:int, ?a:float) =
        Weight.kaiming(Int fanIn, Int fanOut, ?a=a)

    /// <summary>TBD</summary>
    static member uniform(shape:Shape, k:float) =
        -k + dsharp.rand(shape) * 2*k
    
    /// <summary>TBD</summary>
    static member uniform(shape:seq<int>, k:float) = Weight.uniform (Shape shape, k)
    
    /// <summary>TBD</summary>
    static member uniform(shape:seq<Int>, k:float) = Weight.uniform (Shape shape, k)

/// <summary>A model that applies a linear transformation to the incoming data: \(y = xA^T + b\)</summary>
type Linear(inFeatures:Int, outFeatures:Int, ?bias:bool) =
    inherit Model()
    let hasBias = defaultArg bias true
    let w = Parameter(Weight.kaiming(inFeatures, outFeatures))
    let k = 1./sqrt (float outFeatures.ValueOrOne)
    let b = Parameter(if hasBias then Weight.uniform([|outFeatures|], k) else dsharp.tensor([]))
    do base.add([w;b],["Linear-weight";"Linear-bias"])
    
    /// <summary>TBD</summary>
    member _.weight = w

    /// <summary>TBD</summary>
    member _.bias = b

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Linear(%A, %A)" inFeatures outFeatures

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.matmul(value, w.value)
        if hasBias then f + b.value else f
        
    /// <summary>TBD</summary>
    new (inFeatures: int, outFeatures: int, ?bias:bool) =
       Linear(Int inFeatures, Int outFeatures, ?bias=bias)

/// <summary>A model that applies a 1D convolution over an input signal composed of several input planes</summary>
type Conv1d(inChannels:Int, outChannels:Int, kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSize|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv1d-weight";"Conv1d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
        Conv1d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?bias=bias)

/// <summary>A model that applies a 2D convolution over an input signal composed of several input planes</summary>
type Conv2d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv2d-weight";"Conv2d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
        Conv2d(Int inChannels, Int outChannels, ?kernelSize=optInt kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias)

/// <summary>A model that applies a 3D convolution over an input signal composed of several input planes</summary>
type Conv3d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels.ValueOrOne*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv3d-weight";"Conv3d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
        Conv3d(Int inChannels, Int outChannels, ?kernelSize=optInt kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias)


/// <summary>A model that applies a 1D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose1d(inChannels:Int, outChannels:Int, kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?bias:bool, ?outputPadding: Int) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSize|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose1d-weight";"ConvTranspose1d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool, ?outputPadding: int) =
        ConvTranspose1d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?bias=bias, ?outputPadding=optInt outputPadding)

/// <summary>A model that applies a 2D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose2d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?outputPadding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool, ?outputPaddings:seq<Int>) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose2d-weight";"ConvTranspose2d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?outputPadding=outputPadding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations, ?outputPaddings=outputPaddings)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding: int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputPaddings: seq<int>, ?dilations:seq<int>, ?bias:bool) =
        ConvTranspose2d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?outputPadding=optInt outputPadding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias, ?outputPaddings=optInts outputPaddings)

/// <summary>A model that applies a 3D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose3d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?outputPaddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose3d-weight";"ConvTranspose3d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding: int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings: seq<int>, ?bias:bool) =
        ConvTranspose3d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?outputPadding=optInt outputPadding, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?outputPaddings=optInts outputPaddings, ?bias=bias)

/// <summary>A model which during training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.</summary>
type Dropout(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Dropout()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout(?p=p) else value


/// <summary>A model which during training, randomly zero out entire channels. Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
type Dropout2d(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Dropout2d()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout2d(?p=p) else value


/// <summary>A model which during training, randomly zero out entire channels. Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
type Dropout3d(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Dropout3d()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout3d(?p=p) else value


/// <summary>Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs with optional additional channel dimension)</summary>
/// <remarks>
///    <para>
///        The mean and standard-deviation are calculated per-dimension over the mini-batches and
///        \(\gamma\( and \(\beta\) are learnable parameter vectors of size \(C\) (where \(C\) is the
///        input size). By default, the elements of \(\gamma\) are set to 1 and the elements of 
///        \(\beta\) are set to 0. The standard-deviation is calculated via the biased estimator,
///        equivalent to <c>dsharp.variance(input, unbiased=False)</c>.
///    </para>
///    <para>
///        Also by default, during training this layer keeps running estimates of its computed mean
///        and variance, which are then used for normalization during evaluation. The running estimates
///        are kept with a default momentum of 0.1.
///    </para>
///    <para>
///       If trackRunningStats is set to False, this layer then does not keep running estimates,
///       and batch statistics are instead used during evaluation time as well.
///    </para>
/// </remarks>
type BatchNorm1d(numFeatures:Int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
    inherit Model()
    let eps = defaultArg eps 1e-5
    let momentum = defaultArg momentum (dsharp.tensor(0.1))
    let affine = defaultArg affine true
    let trackRunningStats = defaultArg trackRunningStats true
    let reversible = defaultArg reversible false
    let w = Parameter <| if affine then dsharp.ones(numFeatures) else dsharp.zero() // gamma
    let b = Parameter <| if affine then dsharp.zeros(numFeatures) else dsharp.zero() // beta
    let _mean = Parameter <| dsharp.zero()
    let _variance = Parameter <| dsharp.zero()
    do base.add([w;b],["BatchNorm1d-weight";"BatchNorm1d-bias"]) // We don't add mean and variance here because they hold running statistics and are not subject to gradient-based optimization

    /// <summary>TBD</summary>
    member _.mean = _mean.value

    /// <summary>TBD</summary>
    member _.variance = _variance.value

    /// <summary>TBD</summary>
    member _.stddev = _variance.value.sqrt()

    /// <summary>TBD</summary>
    member _.weight = w.value

    /// <summary>TBD</summary>
    member _.bias = b.value

    member private _.updateStats (batchMean:Tensor) (batchVariance:Tensor) (n:int) =
        let batchMean = if reversible then batchMean else batchMean.primal
        let batchVariance = if reversible then batchVariance else batchVariance.primal
        _mean.value <- (1 - momentum) * _mean.value + momentum * batchMean
        // PyTorch seems to use unbiased variance (Bessel's correction) for running batchnorm statistics and biased variance for batch statistics. This seems strange and confusing but we adopt the same behavior for the time being.
        // https://github.com/pytorch/pytorch/issues/19902
        // https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/46
        // Here we transform biased variance to unbiased variance for running statistics
        let batchVariance = batchVariance * (float n) / (float n - 1.)
        _variance.value <- (1 - momentum) * _variance.value + momentum * batchVariance

    /// <summary>TBD</summary>
    override _.getString() = sprintf "BatchNorm1d(%A)" numFeatures

    /// <summary>TBD</summary>
    override m.forward(value) =
        if value.dim = 2 then
            if not (value.shapex.[1] =~= numFeatures) then failwithf "Expecting value to have shape NxL (batchSize x numFeatures) where numFeatures=%A, received value with shape %A" numFeatures value.shapex
            let mean, var =
                if m.mode = Mode.Train || (m.mode = Mode.Eval && not trackRunningStats) then
                    value.mean(0), value.variance(0, unbiased=false)
                else
                    _mean.value, _variance.value
            if not value.symbolic && m.mode = Mode.Train && trackRunningStats then 
                let batchSize = value.shape.[0]
                m.updateStats mean var batchSize
            let res = (value - mean) / (var + eps).sqrt()
            if affine then res * w.value + b.value else res
        elif value.dim = 3 then
            if not (value.shapex.[1] =~= numFeatures) then failwithf "Expecting value to have shape NxCxL (batchSize x numFeatures x length) where numFeatures=%A, received value with shape %A" numFeatures value.shapex
            let vt = value.transpose(0,1).view([numFeatures; Int -1])
            let mean, var =
                if m.mode = Mode.Train || (m.mode = Mode.Eval && not trackRunningStats) then
                    vt.mean(1), vt.variance(1, unbiased=false)
                else
                    _mean.value, _variance.value
            if not value.symbolic && m.mode = Mode.Train && trackRunningStats then
                let n = vt.shape.[1]
                m.updateStats mean var n
            let res = (value - mean.view([1I;numFeatures;1I ])) / (var.view([1I;numFeatures;1I]) + eps).sqrt()
            if affine then res * w.value.view([1I;numFeatures;1I]) + b.value.view([1I;numFeatures;1I]) else res
        else failwithf "Expecting value to have shape NxL (batchSize x Length) or NxCxL (batchSize x numChannels x Length), received value with shape %A" value.shapex

    new (numFeatures:int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
        BatchNorm1d(Int numFeatures, ?eps=eps, ?momentum=momentum, ?affine=affine, ?trackRunningStats=trackRunningStats, ?reversible=reversible)

/// <summary>Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with optional additional channel dimension)</summary>
/// <remarks>
///    <para>
///        The mean and standard-deviation are calculated per-dimension over the mini-batches and
///        \(\gamma\( and \(\beta\) are learnable parameter vectors of size \(C\) (where \(C\) is the
///        input size). By default, the elements of \(\gamma\) are set to 1 and the elements of 
///        \(\beta\) are set to 0. The standard-deviation is calculated via the biased estimator,
///        equivalent to <c>dsharp.variance(input, unbiased=False)</c>.
///    </para>
///    <para>
///        Also by default, during training this layer keeps running estimates of its computed mean
///        and variance, which are then used for normalization during evaluation. The running estimates
///        are kept with a default momentum of 0.1.
///    </para>
///    <para>
///       If trackRunningStats is set to False, this layer then does not keep running estimates,
///       and batch statistics are instead used during evaluation time as well.
///    </para>
/// </remarks>
type BatchNorm2d(numFeatures:Int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
    inherit Model()
    let eps = defaultArg eps 1e-5
    let momentum = defaultArg momentum (dsharp.tensor(0.1))
    let affine = defaultArg affine true
    let trackRunningStats = defaultArg trackRunningStats true
    let reversible = defaultArg reversible false
    let w = Parameter <| if affine then dsharp.ones(numFeatures) else dsharp.zero() // gamma
    let b = Parameter <| if affine then dsharp.zeros(numFeatures) else dsharp.zero() // beta
    let _mean = Parameter <| dsharp.zero()
    let _variance = Parameter <| dsharp.zero()
    do base.add([w;b],["BatchNorm2d-weight";"BatchNorm2d-bias"]) // We don't add mean and variance here because they hold running statistics and are not subject to gradient-based optimization

    /// <summary>TBD</summary>
    member _.mean = _mean.value

    /// <summary>TBD</summary>
    member _.variance = _variance.value

    /// <summary>TBD</summary>
    member _.stddev = _variance.value.sqrt()

    /// <summary>TBD</summary>
    member _.weight = w.value

    /// <summary>TBD</summary>
    member _.bias = b.value

    member private _.updateStats (batchMean:Tensor) (batchVariance:Tensor) (n:int) =
        let batchMean = if reversible then batchMean else batchMean.primal
        let batchVariance = if reversible then batchVariance else batchVariance.primal
        _mean.value <- (1 - momentum) * _mean.value + momentum * batchMean
        // PyTorch seems to use unbiased variance (Bessel's correction) for running batchnorm statistics and biased variance for batch statistics. This seems strange and confusing but we adopt the same behavior for the time being.
        // https://github.com/pytorch/pytorch/issues/19902
        // https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/46
        // Here we transform biased variance to unbiased variance for running statistics
        let batchVariance = batchVariance * (float n) / (float n - 1.)
        _variance.value <- (1 - momentum) * _variance.value + momentum * batchVariance

    /// <summary>TBD</summary>
    override _.getString() = sprintf "BatchNorm2d(%A)" numFeatures

    /// <summary>TBD</summary>
    override m.forward(value) =
        if value.dim <> 4 || not (value.shapex.[1] =~= numFeatures) then failwithf "Expecting value to have shape NxCxHxW (batchSize x numFeatures x height x width) where numFeatures=%A, received value with shape %A" numFeatures value.shapex
        let vt = value.transpose(0,1).view([numFeatures;Int -1])
        let mean, var =
            if m.mode = Mode.Train || (m.mode = Mode.Eval && not trackRunningStats) then
                vt.mean(1), vt.variance(1, unbiased=false)
            else
                _mean.value, _variance.value
        if not value.symbolic && m.mode = Mode.Train && trackRunningStats then
            let n = vt.shape.[1]
            m.updateStats mean var n
        let res = (value - mean.view([1I;numFeatures;1I;1I ])) / (var.view([1I;numFeatures;1I;1I]) + eps).sqrt()
        if affine then res * w.value.view([1I;numFeatures;1I;1I]) + b.value.view([1I;numFeatures;1I;1I]) else res

    /// <summary>TBD</summary>
    new (numFeatures:int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
        BatchNorm2d(Int numFeatures, ?eps=eps, ?momentum=momentum, ?affine=affine, ?trackRunningStats=trackRunningStats, ?reversible=reversible)

/// <summary>Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs with optional additional channel dimension)</summary>
/// <remarks>
///    <para>
///        The mean and standard-deviation are calculated per-dimension over the mini-batches and
///        \(\gamma\( and \(\beta\) are learnable parameter vectors of size \(C\) (where \(C\) is the
///        input size). By default, the elements of \(\gamma\) are set to 1 and the elements of 
///        \(\beta\) are set to 0. The standard-deviation is calculated via the biased estimator,
///        equivalent to <c>dsharp.variance(input, unbiased=False)</c>.
///    </para>
///    <para>
///        Also by default, during training this layer keeps running estimates of its computed mean
///        and variance, which are then used for normalization during evaluation. The running estimates
///        are kept with a default momentum of 0.1.
///    </para>
///    <para>
///       If trackRunningStats is set to False, this layer then does not keep running estimates,
///       and batch statistics are instead used during evaluation time as well.
///    </para>
/// </remarks>
type BatchNorm3d(numFeatures:Int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
    inherit Model()
    let eps = defaultArg eps 1e-5
    let momentum = defaultArg momentum (dsharp.tensor(0.1))
    let affine = defaultArg affine true
    let trackRunningStats = defaultArg trackRunningStats true
    let reversible = defaultArg reversible false
    let w = Parameter <| if affine then dsharp.ones(numFeatures) else dsharp.zero() // gamma
    let b = Parameter <| if affine then dsharp.zeros(numFeatures) else dsharp.zero() // beta
    let _mean = Parameter <| dsharp.zero()
    let _variance = Parameter <| dsharp.zero()
    do base.add([w;b],["BatchNorm3d-weight";"BatchNorm3d-bias"]) // We don't add mean and variance here because they hold running statistics and are not subject to gradient-based optimization

    /// <summary>TBD</summary>
    member _.mean = _mean.value

    /// <summary>TBD</summary>
    member _.variance = _variance.value

    /// <summary>TBD</summary>
    member _.stddev = _variance.value.sqrt()

    /// <summary>TBD</summary>
    member _.weight = w.value

    /// <summary>TBD</summary>
    member _.bias = b.value

    member private _.updateStats (batchMean:Tensor) (batchVariance:Tensor) (n:int) =
        let batchMean = if reversible then batchMean else batchMean.primal
        let batchVariance = if reversible then batchVariance else batchVariance.primal
        _mean.value <- (1 - momentum) * _mean.value + momentum * batchMean
        // PyTorch seems to use unbiased variance (Bessel's correction) for running batchnorm statistics and biased variance for batch statistics. This seems strange and confusing but we adopt the same behavior for the time being.
        // https://github.com/pytorch/pytorch/issues/19902
        // https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/46
        // Here we transform biased variance to unbiased variance for running statistics
        let batchVariance = batchVariance * (float n) / (float n - 1.)
        _variance.value <- (1 - momentum) * _variance.value + momentum * batchVariance

    /// <summary>TBD</summary>
    override _.getString() = sprintf "BatchNorm3d(%A)" numFeatures

    /// <summary>TBD</summary>
    override m.forward(value) =
        if value.dim <> 5 || not (value.shapex.[1] =~= numFeatures) then failwithf "Expecting value to have shape NxCxDxHxW (batchSize x numFeatures x depth x height x width) where numFeatures=%A, received value with shape %A" numFeatures value.shapex
        let vt = value.transpose(0,1).view([numFeatures; Int -1])
        let mean, var =
            if m.mode = Mode.Train || (m.mode = Mode.Eval && not trackRunningStats) then
                vt.mean(1), vt.variance(1, unbiased=false)
            else
                _mean.value, _variance.value
        if not value.symbolic && m.mode = Mode.Train && trackRunningStats then
            let n = vt.shape.[1]
            m.updateStats mean var n
        let res = (value - mean.view([1I;numFeatures;1I;1I;1I])) / (var.view([1I;numFeatures;1I;1I;1I]) + eps).sqrt()
        if affine then res * w.value.view([1I;numFeatures;1I;1I;1I]) + b.value.view([1I;numFeatures;1I;1I; 1I]) else res        

    /// <summary>TBD</summary>
    new (numFeatures:int, ?eps:double, ?momentum:Tensor, ?affine:bool, ?trackRunningStats:bool, ?reversible:bool) =
        BatchNorm3d(Int numFeatures, ?eps=eps, ?momentum=momentum, ?affine=affine, ?trackRunningStats=trackRunningStats, ?reversible=reversible)


/// <summary>Variational Auto-Encoder</summary>
type VAE(xDim:int, zDim:int, ?hDims:seq<int>, ?nonlinearity:Tensor->Tensor, ?nonlinearityLast:Tensor->Tensor) =
    inherit Model()
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let nonlinearity = defaultArg nonlinearity dsharp.relu
    let nonlinearityLast = defaultArg nonlinearityLast dsharp.sigmoid
    let dims =
        if hDims.Length = 0 then
            [|xDim; zDim|]
        else
            Array.append (Array.append [|xDim|] hDims) [|zDim|]
            
    let enc = Array.append [|for i in 0..dims.Length-2 -> Linear(dims.[i], dims.[i+1])|] [|Linear(dims.[dims.Length-2], dims.[dims.Length-1])|]
    let dec = [|for i in 0..dims.Length-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode x =
        let mutable x = x
        for i in 0..enc.Length-3 do
            x <- nonlinearity <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let sampleLatent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode z =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- nonlinearity <| dec.[i].forward(h)
        nonlinearityLast <| dec.[dec.Length-1].forward(h)

    /// <summary>TBD</summary>
    member _.encodeDecode(x:Tensor) =
        let batchSize = x.shape.[0]
        let mu, logVar = encode (x.view([batchSize; xDim]))
        let z = sampleLatent mu logVar
        decode z, mu, logVar

    /// <summary>TBD</summary>
    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    /// <summary>TBD</summary>
    override _.getString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim

    /// <summary>TBD</summary>
    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.viewAs(xRecon), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    /// <summary>TBD</summary>
    member m.loss(x, ?normalize:bool) =
        let normalize = defaultArg normalize true
        let xRecon, mu, logVar = m.encodeDecode x
        let loss = VAE.loss(xRecon, x, mu, logVar)
        if normalize then loss / x.shape.[0] else loss

    /// <summary>TBD</summary>
    member _.sample(?numSamples:int) = 
        let numSamples = defaultArg numSamples 1
        dsharp.randn([|numSamples; zDim|]) |> decode
