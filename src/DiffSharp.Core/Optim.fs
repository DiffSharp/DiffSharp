namespace rec DiffSharp.Optim
open DiffSharp
open DiffSharp.Shorten
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Util


/// <namespacedoc>
///   <summary>Contains types and functionality related to optimizing tensor models and functions.</summary>
/// </namespacedoc>
///
/// <summary>Represents an optimizer.</summary>
[<AbstractClass>]
type Optimizer(model:Model) =

    /// <summary>TBD</summary>
    member val model = model

    /// <summary>TBD</summary>
    member o.step() = model.parametersDict.iter(fun (n, p) -> let t = o.updateRule n p.value in p.value <- t)

    /// <summary>TBD</summary>
    abstract member updateRule: string -> Tensor -> Tensor


/// <summary>TBD</summary>
type SGD(model, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let lr = defaultArg lr (dsharp.tensor(1e-3))
    let nesterov = defaultArg nesterov true
    let reversible = defaultArg reversible false
    let mutable momInit = false
    let mutable momBuffer = ParameterDict()

    /// <summary>TBD</summary>
    override o.updateRule name t = 
        let mutable d = t.derivative
        let t = if reversible then t else t.primal
        match weightDecay with
        | Some wd -> d <- d.add(t.primal * wd)
        | None -> ()
        match momentum with
        | Some mom ->
            if not momInit then 
                momBuffer <- model.parametersDict.map(fun (t:Tensor) -> t.derivative)
                momInit <- true
            let mb = momBuffer.[name]
            let mb = mb.mul(mom).add(d)
            momBuffer.[name] <- mb
            if nesterov then d <- d.add(mb*mom)
            else d <- mb
        | None -> ()   
        t - lr * d


/// <summary>TBD</summary>
type Adam(model, ?lr:Tensor, ?beta1:Tensor, ?beta2:Tensor, ?eps:Tensor, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let lr = defaultArg lr (dsharp.tensor(1e-3))
    let beta1 = defaultArg beta1 (dsharp.tensor(0.9))
    let beta2 = defaultArg beta2 (dsharp.tensor(0.999))
    let eps = defaultArg eps (dsharp.tensor(1e-8))
    let reversible = defaultArg reversible false
    let mutable stateStep = 0
    let mutable stateExpAvg = ParameterDict()
    let mutable stateExpAvgSq = ParameterDict()

    /// <summary>TBD</summary>
    override o.updateRule name t =
        let mutable d = t.derivative
        let t = if reversible then t else t.primal
        match weightDecay with
        | Some wd -> d <- d.add(t.primal * wd)
        | None -> ()
        if stateStep = 0 then
            stateExpAvg <- model.parametersDict.map(fun (t:Tensor) -> t.zerosLike())
            stateExpAvgSq <- model.parametersDict.map(fun (t:Tensor) -> t.zerosLike())
        stateStep <- stateStep + 1
        let expAvg = stateExpAvg.[name].mul(beta1).add(d*(1.-beta1))
        let expAvgSq = stateExpAvgSq.[name].mul(beta2).add(d*d*(1.-beta2))
        stateExpAvg.[name] <- expAvg
        stateExpAvgSq.[name] <- expAvgSq
        let biasCorrection1 = 1. - beta1 ** stateStep
        let biasCorrection2 = 1. - beta2 ** stateStep
        let denom = (expAvgSq.sqrt() / biasCorrection2.sqrt()).add(eps)
        let stepSize = lr / biasCorrection1
        t - stepSize * (expAvg/denom)


/// <summary>TBD</summary>
type optim =

    static member internal optimizeFun(update:Tensor->Tensor*Tensor, x0:Tensor, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let iters = defaultArg iters -1
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
        let mutable fxMin = System.Double.MaxValue
        let mutable fxMax = System.Double.MinValue
        let mutable fxPrev = System.Double.MinValue    
        let mutable i = -1
        let mutable stop = false
        let start = System.DateTime.Now
        if print then printfn "Duration   |Iters| Value"
        while not stop do
            i <- i + 1
            let nfx, nx = update x
            fx <- nfx
            let fxScalar = float fx

            if fx.hasnan() || fx.hasinf() then
                status <- "Diverged"
                stop <- true
            elif thresholdGiven && fxScalar <= threshold then
                status <- sprintf "Converged (value < %g)" threshold
                stop <- true
            elif (iters <> -1) && (i=iters-1) then
                status <- sprintf "Iters=%d reached" iters
                stop <- true
            elif fxScalar < fxMin then
                fxMin <- fxScalar
                status <- "- New min"
            elif fxScalar > fxMax then
                fxMax <- fxScalar
                status <- "+ New max"
            elif fxScalar < fxPrev then
                status <- "-"
            elif fxScalar > fxPrev then
                status <- "+"
            else
                status <- ""

            let duration = System.DateTime.Now - start
            if print && ((i+1) % printEvery = 0 || i = 0 || stop) then
                let printDepthPrefix = String.replicate nx.depth "  "
                let durationStr = duration.ToString(@"d\.hh\:mm\:ss")
                printfn "%s%s%s | %3d | %e %s%s" printDepthPrefix printPrefix durationStr (i+1) fxScalar status printPostfix

            fxPrev <- fxScalar
            if not stop then x <- nx
        fx, x

    /// <summary>TBD</summary>
    static member internal optimizeModel(model:Model, optimizer:Optimizer, dataloader:DataLoader, loss:Tensor->Tensor->Tensor, ?iters:int, ?epochs:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let iters, epochs =
            match iters, epochs with
            | Some _, Some _ -> failwithf "Expecting only one of iters, epochs"
            | Some i, None -> i, -1
            | None, Some e -> -1, e
            | None, None -> -1, -1
        let threshold, thresholdGiven = 
            match threshold with
            | Some t -> t, true
            | None -> -1., false
        let print = defaultArg print true
        let printEvery = defaultArg printEvery (max 1 (iters/20))
        let printPrefix = defaultArg printPrefix ""
        let printPostfix = defaultArg printPostfix ""
        let mutable status = ""
        let mutable epoch = -1
        let mutable i = -1
        let mutable lMin = System.Double.MaxValue
        let mutable lMax = System.Double.MinValue
        let mutable lPrev = System.Double.MinValue
        let mutable stop = false
        let start = System.DateTime.Now
        if print then printfn "Duration   |Iters| Ep|%s| Loss" (stringPadAs "Minib" (sprintf " 1/%d " dataloader.length))
        while not stop do
            epoch <- epoch + 1
            dataloader.epoch() 
            |> Seq.takeWhile (fun (bi, data, targets) ->
                i <- i + 1
                model.reverseDiff()
                let o = model.forward(data)
                let l = loss targets o
                l.reverse()
                optimizer.step()
                
                let lScalar = float l

                if l.hasnan() || l.hasinf() then
                    status <- "Diverged"
                    stop <- true
                elif thresholdGiven && lScalar <= threshold then
                    status <- sprintf "Converged (loss < %g)" threshold
                    stop <- true
                elif (iters <> -1) && (i=iters-1) then
                    status <- sprintf "Iters=%d reached" iters
                    stop <- true
                elif lScalar < lMin then
                    lMin <- lScalar
                    status <- "- New min"
                elif lScalar > lMax then
                    lMax <- lScalar
                    status <- "+ New max"
                elif (epochs <> -1) && (epoch=epochs) then
                    status <- sprintf "Epochs=%d reached" epochs
                    stop <- true
                elif lScalar < lPrev then
                    status <- "-"
                elif lScalar > lPrev then
                    status <- "+"
                else
                    status <- ""

                if print && ((i+1) % printEvery = 0 || i = 0 || stop) then
                    let duration = System.DateTime.Now - start
                    let durationStr = duration.ToString(@"d\.hh\:mm\:ss")
                    printf "%s%s | %3d | %d | %d/%d | %e %s%s" printPrefix durationStr (i+1) (epoch+1) (bi+1) dataloader.length lScalar status printPostfix
                lPrev <- lScalar
                not stop
            ) |> Seq.iter ignore

    /// <summary>TBD</summary>
    static member sgd(f, x0:Tensor, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
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
        optim.optimizeFun(update, x0, ?iters=iters, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix)

    /// <summary>TBD</summary>
    static member adam(f, x0:Tensor, ?lr:Tensor, ?beta1:Tensor, ?beta2:Tensor, ?eps:Tensor, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let lr = defaultArg lr (dsharp.tensor(1e-3))
        let beta1 = defaultArg beta1 (dsharp.tensor(0.9))
        let beta2 = defaultArg beta2 (dsharp.tensor(0.999))
        let eps = defaultArg eps (dsharp.tensor(1e-8))
        let mutable step = 0
        let mutable expAvg = x0.zerosLike()
        let mutable expAvgSq = x0.zerosLike()
        let update x =
            let f, g = dsharp.fg f x
            step <- step + 1
            expAvg <- expAvg.mul(beta1).add(g*(1.-beta1))
            expAvgSq <- expAvgSq.mul(beta2).add(g*g*(1.-beta2))
            let biasCorrection1 = 1. - beta1 ** step
            let biasCorrection2 = 1. - beta2 ** step
            let denom = (expAvgSq.sqrt() / biasCorrection2.sqrt()).add(eps)
            let p = expAvg / denom
            let stepSize = lr / biasCorrection1
            f, x - stepSize * p
        optim.optimizeFun(update, x0, ?iters=iters, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix)

    /// <summary>TBD</summary>
    static member sgd(model, dataloader, loss, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool, ?iters:int, ?epochs:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let optimizer = SGD(model, ?lr=lr, ?momentum=momentum, ?nesterov=nesterov, ?weightDecay=weightDecay, ?reversible=reversible)
        optim.optimizeModel(model, optimizer, dataloader, loss, ?iters=iters, ?epochs=epochs, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix)

    /// <summary>TBD</summary>
    static member adam(model, dataloader, loss, ?lr:Tensor, ?beta1:Tensor, ?beta2:Tensor, ?eps:Tensor, ?weightDecay:Tensor, ?reversible:bool, ?iters:int, ?epochs:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string) =
        let optimizer = Adam(model, ?lr=lr, ?beta1=beta1, ?beta2=beta2, ?eps=eps, ?weightDecay=weightDecay, ?reversible=reversible)
        optim.optimizeModel(model, optimizer, dataloader, loss, ?iters=iters, ?epochs=epochs, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix)
