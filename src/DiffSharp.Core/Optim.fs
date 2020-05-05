namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model

[<AbstractClass>]
type Optimizer(model:Model) =
    member val model = model
    abstract member updateRule: string -> Tensor -> Tensor
    member o.step() = model.Parameters.iter(fun (n, p) -> let t = o.updateRule n p.value in p.value <- t)


type SGD(model, learningRate:Tensor, ?momentum:Tensor, ?dampening:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let lr = learningRate
    let mutable momBuffer = ParameterDict()
    let mutable momInit = false
    let dampening = defaultArg dampening (lr.zeroLike())
    let nesterov = defaultArg nesterov true
    let reversible = defaultArg reversible false
    override o.updateRule name t = 
        let mutable d = t.derivative
        match weightDecay with
        | Some wd -> d <- d.add(t.primal * wd)
        | None -> ()
        match momentum with
        | Some mom ->
            if not momInit then 
                momBuffer <- model.Parameters.map(fun (t:Tensor) -> t.derivative)
                momInit <- true
            let mb = momBuffer.[name] 
            let mb = mb.mul(mom).add(d*(1.-dampening))
            momBuffer.[name] <- mb
            if nesterov then d <- d.add(mb*mom)
            else d <- mb
        | None -> ()   
        if reversible then
            t - lr * d
        else
            t.primal - lr * d