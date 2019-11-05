namespace DiffSharp.NN
open DiffSharp
open System.Collections.Generic

type Parameter(tensor:Tensor) =
    let mutable t = tensor
    member p.Tensor 
        with get() = t
        and set(tensor) = t <- tensor
    member p.ForwardDiff(derivative) = t <- t.ForwardDiff(derivative); t
    member p.ReverseDiff() = t <- t.ReverseDiff(); t
    member p.NoDiff() = t <- t.NoDiff(); t
    override p.ToString() = sprintf "Parameter %A" t

[<AbstractClass>]
type Layer() =
    member val Parameters:Dictionary<string, Parameter> = Dictionary()
    member inline l.AddParameters(parameters:list<string * 'a>) =
        for n, p in parameters do 
            match (box p) with
            | :? Parameter as p -> l.Parameters.Add(n, p)
            | :? Layer as layer ->
                for KeyValue(nn, pp) in layer.Parameters do
                    l.Parameters.Add(n + "_" + nn, pp)
            | _ -> failwithf "Unsupported type. Expecting a list<string * 'a> where 'a is Layer or Parameter"
    member l.IterParameters(f) =
        let keys = Array.create l.Parameters.Count ""
        l.Parameters.Keys.CopyTo(keys, 0)
        for k in keys do f k l.Parameters.[k]
    member l.ForwardDiff(derivatives:Dictionary<string, Tensor>) = l.IterParameters(fun k p -> p.ForwardDiff(derivatives.[k]) |> ignore)
    member l.ReverseDiff() = l.IterParameters(fun _ p -> p.ReverseDiff() |> ignore)
    member l.NoDiff() = l.IterParameters(fun _ p -> p.NoDiff() |> ignore)
    abstract member Forward: Tensor -> Tensor
    
type Linear(inFeatures, outFeatures) =
    inherit Layer()
    let w = Parameter(Tensor.RandomNormal([inFeatures; outFeatures]))
    let b = Parameter(Tensor.RandomNormal([outFeatures]))
    do base.AddParameters(["weight", w; "bias", b])
    override l.Forward(value) = Tensor.MatMul(value, w.Tensor) + b.Tensor
