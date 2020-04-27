namespace DiffSharp.Model
open DiffSharp
open DiffSharp.Util
open System.Collections.Generic

type Parameter =
    val mutable value:Tensor
    new(value) = {value=value}
    member p.forwardDiff(derivative:Tensor) = p.value <- p.value.forwardDiff(derivative)
    member p.reverseDiff() = p.value <- p.value.reverseDiff()
    member p.noDiff() = p.value <- p.value.noDiff()
    override p.ToString() = sprintf "Parameter(shape: %A, value: %A)" p.value.shape p.value

type ParameterDict() =
    member val values:Dictionary<string, Parameter> = Dictionary()
    member d.Item
        with get key = d.values.[key].value
        and set key v = d.values.[key].value <- v
    member d.add(name, parameter) = d.values.Add(name, parameter)
    member d.add(parameters:list<string*Parameter>) = for (n, p) in parameters do d.add(n, p)
    member d.add(parameters:ParameterDict) = for KeyValue(n, p) in parameters.values do d.add(n, p)
    member d.copy() = d.map(fun (t:Tensor) -> t)
    member d.map(f:string*Parameter->string*Parameter) =
        let ret = ParameterDict()
        for KeyValue(n, p) in d.values do ret.values.Add(f(n,p))
        ret
    member d.map(f:string*Tensor->string*Tensor) = d.map(fun (n, p:Parameter) -> let nn, tt = f(n, p.value) in nn, Parameter(tt))
    member d.map(f:Tensor->Tensor) = d.map(fun (n,t) -> n, f t)
    member d.set(parameters:ParameterDict) = d.iter(fun (n, p) -> p.value <- parameters.[n])
    member d.iter(f:string*Parameter->unit) = for KeyValue(n, p) in d.values do f(n,p)
    member d.forwarddiff(derivatives:ParameterDict) = d.iter(fun (n, p) -> p.forwardDiff(derivatives.[n]))
    member d.reverseDiff() = d.iter(fun (_, p) -> p.reverseDiff())
    member d.noDiff() = d.iter(fun (_, p) -> p.noDiff())
    member d.flatten() =
        let ts = [for t in d.values.Values do t.value.view(-1)]
        dsharp.cat(ts)
    member d.unflatten(tensors:Tensor) =
        let shapes = [|for t in d.values.Values do t.value.shape|]
        let sizes = [|for s in shapes do shapeLength s|]
        let ts = Array.map2 (fun (t:Tensor) (s:int[]) -> t.view(s)) (tensors.split(sizes)) shapes
        let mutable i = 0
        let keys = getKeys d.values
        for n in keys do
            d.[n] <- ts.[i]
            i <- i+1
    member d.unflattenToNew(tensors:Tensor) = 
        let dd = d.copy()
        dd.unflatten(tensors)
        dd

[<AbstractClass>]
type Model() =
    member val Parameters:ParameterDict = ParameterDict()
    member val SubModels:Dictionary<string, Model> = Dictionary()
    member inline m.add(parameters:list<string * 'a>) =
        for n, p in parameters do
            match (box p) with
            | :? Parameter as p -> 
                m.Parameters.add(n, p)
            | :? Model as mm ->
                m.SubModels.Add(n, mm)
                m.Parameters.add(mm.Parameters.map(fun (nn, pp:Parameter) -> (n + "__" + nn, pp)))
            | _ -> failwithf "Unsupported type. Expecting a list<string * 'a> where 'a is Parameter or Model"
    member m.forwardDiff(derivatives:ParameterDict) = m.Parameters.forwarddiff(derivatives)
    member m.reverseDiff() = m.Parameters.reverseDiff()
    member m.noDiff() = m.Parameters.noDiff()
    member m.setParameters(parameters:ParameterDict) = m.Parameters.set(parameters)
    member m.setParameters(parameters:Tensor) = m.Parameters.unflatten(parameters)
    member m.getParameters() = m.Parameters.flatten()
    member m.nparameters() = m.Parameters.flatten().nelement
    member m.forwardParams (input:Tensor) (parameters:Tensor) =
        m.setParameters(parameters)
        m.forward(input)
    member m.forwardCompose (f:Tensor->Tensor) (input:Tensor) (parameters:Tensor) =
        m.forwardParams input parameters |> f
    member m.forwardLoss (f:Tensor->Tensor->Tensor) (input:Tensor) (target:Tensor) (parameters:Tensor) =
        m.forwardCompose (f target) input parameters
    abstract member forward: Tensor -> Tensor


type Init() =
    static member kaiming(fanIn, fanOut, ?a:float) = 
        let a = defaultArg a (sqrt 5.)
        let w = dsharp.randn([fanIn; fanOut])
        let s = sqrt (2. / ((1. + a*a) * (float fanIn)))
        w * s
    static member init(shape:int[], k:float) =
        -k + dsharp.rand(shape) * 2*k


type Linear(inFeatures, outFeatures, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./sqrt (float outFeatures)
    let w = Parameter(Init.kaiming(inFeatures, outFeatures))
    let b = Parameter(if bias then Init.init([|outFeatures|], k) else dsharp.zero())
    do base.add(["weight", w; "bias", b])
    override l.forward(value) =
        let f = dsharp.matmul(value, w.value)
        if bias then f + b.value else f


type Conv1d(inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
    inherit Model()
    let stride = defaultArg stride 1
    let padding = defaultArg padding 0
    let dilation = defaultArg dilation 1
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize))
    let w = Parameter <| Init.init([|outChannels; inChannels; kernelSize|], k)
    let b = Parameter <| if bias then Init.init([|outChannels|], k) else dsharp.zero()
    do base.add(["weight", w; "bias", b])
    override c.forward(value) =
        let f = dsharp.conv1d(value, w.value, stride=stride, padding=padding, dilation=dilation)
        if bias then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1]) else f


type Conv2d(inChannels:int, outChannels:int, kernelSize:seq<int>, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSize = kernelSize |> Array.ofSeq
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize.[0]*kernelSize.[1]))
    let w = Parameter <| Init.init([|outChannels; inChannels; kernelSize.[0]; kernelSize.[1]|], k)
    let b = Parameter <| if bias then Init.init([|outChannels|], k) else dsharp.zero()
    do base.add(["weight", w; "bias", b])
    override c.forward(value) =
        let f = dsharp.conv2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1]) else f