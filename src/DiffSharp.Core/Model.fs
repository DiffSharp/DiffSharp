namespace DiffSharp.Model
open DiffSharp
open DiffSharp.Util
open System.Collections.Generic


type TensorDict() =
    member val Tensors:Dictionary<string, Tensor> = Dictionary()
    member d.Item
        with get key = d.Tensors.[key]
        and set key value = d.Tensors.[key] <- value
    member d.add(name, tensor) = d.Tensors.Add(name, tensor)
    member d.map(f) =
        let ret = TensorDict()
        for KeyValue(n, t) in d.Tensors do ret.add(n, f n t)
        ret
    member d.copy() = d.map(fun _ t -> t)
    member d.forwardDiff(derivatives:TensorDict) = d.map(fun n t -> t.forwardDiff(derivatives.[n]))
    member d.reverseDiff() = d.map(fun _ t -> t.reverseDiff())
    member d.noDiff() = d.map(fun _ t -> t.noDiff())
    member d.flatten() =
        let ts = [for t in d.Tensors.Values do t.view(-1)]
        dsharp.cat(ts)
    member d.unflatten(tensors:Tensor) =
        let ret = TensorDict()
        let shapes = [|for t in d.Tensors.Values do t.shape|]
        let sizes = [|for s in shapes do shapeLength s|]
        let ts = Array.map2 (fun (t:Tensor) (s:int[]) -> t.view(s)) (tensors.split(sizes)) shapes
        let mutable i = 0
        let keys = getKeys d.Tensors
        for n in keys do
            ret.add(n, ts.[i])
            i <- i+1
        ret

[<AbstractClass>]
type Model() =
    member val Parameters = TensorDict()
    member val SubModels:Dictionary<string, Model> = Dictionary()
    member inline m.addParameters(parameters:list<string * 'a>) =
        for n, p in parameters do
            match (box p) with
            | :? Tensor as t -> 
                // printfn "adding Tensor %A %A" n t
                m.Parameters.add(n, t)
            | :? Model as model ->
                // printfn "adding submodel %A" model
                m.SubModels.[n] <- model
                for KeyValue(nn, tt) in model.Parameters.Tensors do
                    // printfn "adding Tensor %A %A" (n + "__" + nn) tt
                    m.Parameters.add(n + "__" + nn, tt)
            | _ -> failwithf "Unsupported type. Expecting a list<string * 'a> where 'a is Tensor or Model"
    member m.setParameters(parameters:TensorDict) =
        let keys = getKeys m.Parameters.Tensors
        for n in keys do
            m.Parameters.[n] <- parameters.[n]
        for KeyValue(n, model) in m.SubModels do
            let keys = getKeys model.Parameters.Tensors
            for nn in keys do
                model.Parameters.Tensors.[nn] <- m.Parameters.[n + "__" + nn]
    member m.setParameters(parameters:Tensor) = m.setParameters(m.Parameters.unflatten(parameters))
    member m.getParameters() = m.Parameters.flatten()
    member m.forwardDiff(derivatives) = m.setParameters(m.Parameters.forwardDiff(derivatives))
    member m.reverseDiff() = m.setParameters(m.Parameters.reverseDiff())
    member m.noDiff() = m.setParameters(m.Parameters.noDiff())
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
    static member bias(fanOut) =
        let b = 1./sqrt (float fanOut)
        -b + dsharp.rand([fanOut]) * 2*b

type Linear(inFeatures, outFeatures, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let w = Init.kaiming(inFeatures, outFeatures)
    let b = if bias then Init.bias(outFeatures) else dsharp.zero()
    do base.addParameters(["weight", w; "bias", b])
    override l.forward(value) =
        let w, b = l.Parameters.["weight"], l.Parameters.["bias"]
        let f = dsharp.matmul(value, w)
        if bias then f + b else f


type Conv1d(inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
    inherit Model()
    let stride = defaultArg stride 1
    let padding = defaultArg padding 0
    let dilation = defaultArg dilation 1
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize))
    let w = -k + dsharp.rand([outChannels; inChannels; kernelSize]) * 2*k
    let b = if bias then -k + dsharp.rand(outChannels) * 2*k else dsharp.zero()
    do base.addParameters(["weight", w; "bias", b])
    override c.forward(value) =
        let w, b = c.Parameters.["weight"], c.Parameters.["bias"]
        let f = dsharp.conv1d(value, w, stride=stride, padding=padding, dilation=dilation)
        if bias then f + b.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1]) else f


type Conv2d(inChannels:int, outChannels:int, kernelSize:seq<int>, ?stride:seq<int>, ?padding:seq<int>, ?dilation:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSize = kernelSize |> Array.ofSeq
    let stride = defaultArg stride (seq [1; 1]) |> Array.ofSeq
    let padding = defaultArg padding (seq [0; 0]) |> Array.ofSeq
    let dilation = defaultArg dilation (seq [1; 1]) |> Array.ofSeq
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize.[0]*kernelSize.[1]))
    let w = -k + dsharp.rand([outChannels; inChannels; kernelSize.[0]; kernelSize.[1]]) * 2*k
    let b = if bias then -k + dsharp.rand(outChannels) * 2*k else dsharp.zero()
    do base.addParameters(["weight", w; "bias", b])
    new(inChannels:int, outChannels:int, kernelSize:int) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [1; 1], [0; 0], [1; 1], true)
    new(inChannels:int, outChannels:int, kernelSize:int, bias:bool) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [1; 1], [0; 0], [1; 1], bias)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [0; 0], [1; 1], true)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int, bias:bool) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [0; 0], [1; 1], bias)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int, padding:int) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [padding; padding], [1; 1], true)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int, padding:int, bias:bool) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [padding; padding], [1; 1], bias)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int, padding:int, dilation:int) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [padding; padding], [dilation; dilation], true)
    new(inChannels:int, outChannels:int, kernelSize:int, stride:int, padding:int, dilation:int, bias:bool) =
        Conv2d(inChannels, outChannels, [kernelSize; kernelSize], [stride; stride], [padding; padding], [dilation; dilation], bias)
    override c.forward(value) =
        let w, b = c.Parameters.["weight"], c.Parameters.["bias"]
        let f = dsharp.conv2d(value, w, stride=stride, padding=padding, dilation=dilation)
        if bias then f + b.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1]) else f