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


type Linear(inFeatures, outFeatures) =
    inherit Model()
    let w = Init.kaiming(inFeatures, outFeatures)
    let b = Init.bias(outFeatures)
    do base.addParameters(["weight", w; "bias", b])
    override l.forward(value) =
        dsharp.matmul(value, l.Parameters.["weight"])
        + l.Parameters.["bias"]