namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model

[<AbstractClass>]
type Optimizer(model:Model) =
    member val model = model
    abstract member parameterUpdate: string -> Tensor -> Tensor
    member o.step() = o.model.updateParameters(o.model.Parameters.map(o.parameterUpdate))


type SGD(model, ?learningRate:Tensor, ?momentum:Tensor, ?nesterov:bool) =
    inherit Optimizer(model)
    let lr = defaultArg learningRate (dsharp.tensor(0.0001))
    let mom = momentum
    let nesterov = defaultArg nesterov true
    // let momentumBuffer = TensorDict()
    override o.parameterUpdate _ t = 
        match mom with
        | Some _ -> 
            if nesterov then failwith "not implemented"
            else failwith "not implemented"
        | None -> t.primal - lr * t.derivative