namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model


[<AbstractClass>]
type Optimizer(model:Model) =
    member val model = model
    member o.step() = model.Parameters.iter(fun (n, p) -> let t = o.updateRule n p.value in p.value <- t)
    abstract member updateRule: string -> Tensor -> Tensor
    static member internal optimizeIters(update:Tensor->Tensor*Tensor, x0:Tensor, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let iters = defaultArg iters 50
        let threshold, thresholdGiven = 
            match threshold with
            | Some t -> t, true
            | None -> -1., false
        let print = defaultArg print true
        let printEvery = defaultArg printEvery (max 1 (iters/20))
        let printPrefix = defaultArg printPrefix ""
        let printPostfix = defaultArg printPostfix ""
        let mutable status = ""
        let mutable x = x0
        let mutable fx = dsharp.zero()
        let mutable i = 0
        let mutable stop = false
        while not stop do
            i <- i + 1
            let nfx, nx = update x
            fx <- nfx
            let fxScalar = fx.toScalar() |> System.Convert.ToDouble

            if fx.hasnan() || fx.hasinf() then
                status <- "DIVERGED"
                stop <- true
            elif thresholdGiven && fxScalar <= threshold then
                status <- "CONVERGED"
                stop <- true
            elif i=iters then
                status <- "ITERS REACHED"
                stop <- true

            if print && ((i+1) % printEvery = 0 || i = iters-1 || stop) then
                let printDepth = String.replicate nx.depth "  "
                printfn "%s%s%A/%A | %g %s%s" printDepth printPrefix (i+1) iters fxScalar status printPostfix

            if not stop then x <- nx
        fx, x
    static member sgd(f, x0:Tensor, ?iters:int, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let lr = defaultArg lr (dsharp.tensor(0.001))
        let mutable momBuffer = dsharp.zero()
        let mutable momInit = false
        let nesterov = defaultArg nesterov true
        let mutable p = dsharp.zero()
        let update x =
            let f, g = dsharp.fg f x
            p <- g
            match momentum with
            | Some mom ->
                if not momInit then momBuffer <- g; momInit <- true
                momBuffer <- momBuffer.mul(mom).add(g)
                if nesterov then p <- p.add(momBuffer*mom)
                else p <- momBuffer
            | None -> ()
            f, x - lr * p
        Optimizer.optimizeIters(update, x0, ?iters=iters, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix)


type SGD(model, lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let mutable momBuffer = ParameterDict()
    let mutable momInit = false
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
            let mb = mb.mul(mom).add(d)
            momBuffer.[name] <- mb
            if nesterov then d <- d.add(mb*mom)
            else d <- mb
        | None -> ()   
        if reversible then
            t - lr * d
        else
            t.primal - lr * d