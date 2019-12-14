namespace DiffSharp.Model
open DiffSharp
open DiffSharp.Util
open System.Collections.Generic


type TensorDict() =
    member val Tensors:Dictionary<string, Tensor> = Dictionary()
    member d.Item
        with get key = d.Tensors.[key]
        and set key value = d.Tensors.[key] <- value
    member d.Add(name, tensor) = d.Tensors.Add(name, tensor)
    member d.Map(f) =
        let ret = TensorDict()
        for KeyValue(n, t) in d.Tensors do ret.Add(n, f n t)
        ret
    member d.ForwardDiff(derivatives:TensorDict) = d.Map(fun n t -> t.ForwardDiff(derivatives.[n]))
    member d.ReverseDiff() = d.Map(fun _ t -> t.ReverseDiff())
    member d.NoDiff() = d.Map(fun _ t -> t.NoDiff())

[<AbstractClass>]
type Model() =
    member val Parameters = TensorDict()
    member val SubModels:Dictionary<string, Model> = Dictionary()
    member inline m.AddParameters(parameters:list<string * 'a>) =
        for n, p in parameters do
            match (box p) with
            | :? Tensor as t -> 
                // printfn "adding Tensor %A %A" n t
                m.Parameters.Add(n, t)
            | :? Model as model ->
                // printfn "adding submodel %A" model
                m.SubModels.[n] <- model
                for KeyValue(nn, tt) in model.Parameters.Tensors do
                    printfn "adding Tensor %A %A" (n + "__" + nn) tt
                    m.Parameters.Add(n + "__" + nn, tt)
            | _ -> failwithf "Unsupported type. Expecting a list<string * 'a> where 'a is Tensor or Model"
    member m.UpdateParameters(parameters:TensorDict) =
        let keys = getKeys m.Parameters.Tensors
        for n in keys do
            m.Parameters.[n] <- parameters.[n]
        for KeyValue(n, model) in m.SubModels do
            let keys = getKeys model.Parameters.Tensors
            for nn in keys do
                model.Parameters.Tensors.[nn] <- m.Parameters.[n + "__" + nn]
    member m.ForwardDiff(derivatives) = m.UpdateParameters(m.Parameters.ForwardDiff(derivatives))
    member m.ReverseDiff() = m.UpdateParameters(m.Parameters.ReverseDiff())
    member m.NoDiff() = m.UpdateParameters(m.Parameters.NoDiff())
    abstract member Forward: Tensor -> Tensor


type Linear(inFeatures, outFeatures) =
    inherit Model()
    let w = Tensor.RandomNormal([inFeatures; outFeatures])
    let b = Tensor.RandomNormal([outFeatures])
    do base.AddParameters(["weight", w; "bias", b])
    override l.Forward(value) =
        Tensor.MatMul(value, l.Parameters.["weight"])
        + l.Parameters.["bias"]
