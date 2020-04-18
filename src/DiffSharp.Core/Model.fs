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
    member d.forwardDiff(derivatives:TensorDict) = d.map(fun n t -> t.forwardDiff(derivatives.[n]))
    member d.reverseDiff() = d.map(fun _ t -> t.reverseDiff())
    member d.noDiff() = d.map(fun _ t -> t.noDiff())

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
    member m.UpdateParameters(parameters:TensorDict) =
        let keys = getKeys m.Parameters.Tensors
        for n in keys do
            m.Parameters.[n] <- parameters.[n]
        for KeyValue(n, model) in m.SubModels do
            let keys = getKeys model.Parameters.Tensors
            for nn in keys do
                model.Parameters.Tensors.[nn] <- m.Parameters.[n + "__" + nn]
    member m.forwardDiff(derivatives) = m.UpdateParameters(m.Parameters.forwardDiff(derivatives))
    member m.reverseDiff() = m.UpdateParameters(m.Parameters.reverseDiff())
    member m.noDiff() = m.UpdateParameters(m.Parameters.noDiff())
    abstract member forward: Tensor -> Tensor


type Linear(inFeatures, outFeatures) =
    inherit Model()
    let w = dsharp.randn([inFeatures; outFeatures])
    let b = dsharp.randn([outFeatures])
    do base.addParameters(["weight", w; "bias", b])
    override l.forward(value) =
        dsharp.matmul(value, l.Parameters.["weight"])
        + l.Parameters.["bias"]
