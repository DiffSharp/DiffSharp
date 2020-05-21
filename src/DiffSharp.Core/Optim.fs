namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model
open DiffSharp.Data


[<AbstractClass>]
type Optimizer(model:Model) =
    member val model = model
    member o.step() = model.Parameters.iter(fun (n, p) -> let t = o.updateRule n p.value in p.value <- t)
    abstract member updateRule: string -> Tensor -> Tensor
    static member internal optimizeFun(update:Tensor->Tensor*Tensor, x0:Tensor, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string, ?printNewLine:bool) =
        let iters = defaultArg iters 50
        let threshold, thresholdGiven = 
            match threshold with
            | Some t -> t, true
            | None -> -1., false
        let print = defaultArg print true
        let printEvery = defaultArg printEvery 1 // (max 1 (iters/20))
        let printPrefix = defaultArg printPrefix ""
        let printPostfix = defaultArg printPostfix ""
        let printNewLine = defaultArg printNewLine true
        let mutable printEnd = if printNewLine then "\n" else "                    \r"
        let mutable status = ""
        let mutable x = x0
        let mutable fx = dsharp.zero()
        let mutable fxMin = System.Double.MaxValue
        let mutable fxMax = System.Double.MinValue
        let mutable fxPrev = System.Double.MinValue    
        let mutable i = -1
        let mutable stop = false
        if print then printfn "Iters| Value"
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
            elif i=iters-1 then
                status <- sprintf "Iters=%d reached" iters
                stop <- true
            elif fxScalar < fxMin then
                fxMin <- fxScalar
                status <- "🡾 New min"
            elif fxScalar > fxMax then
                fxMax <- fxScalar
                status <- "🡽 New max"
            elif fxScalar < fxPrev then
                status <- "🡾"
            else
                status <- "🡽"

            if print && ((i+1) % printEvery = 0 || i = 0 || stop) then
                let printDepthPrefix = String.replicate nx.depth "  "
                if stop then printEnd <- "\n"
                printf "%s%s%4d | %e %s%s%s" printDepthPrefix printPrefix (i+1) fxScalar status printPostfix printEnd

            fxPrev <- fxScalar
            if not stop then x <- nx
        fx, x

    static member internal optimizeModel(model:Model, optimizer:Optimizer, dataloader:DataLoader, loss:Tensor->Tensor->Tensor, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string, ?printNewLine:bool) =
        let iters = defaultArg iters 50
        let threshold, thresholdGiven = 
            match threshold with
            | Some t -> t, true
            | None -> -1., false
        let print = defaultArg print true
        let printEvery = defaultArg printEvery 1 // (max 1 (iters/20))
        let printPrefix = defaultArg printPrefix ""
        let printPostfix = defaultArg printPostfix ""
        let printNewLine = defaultArg printNewLine true
        let mutable printEnd = if printNewLine then "\n" else "                    \r"
        let mutable status = ""
        let mutable epoch = -1
        let mutable i = -1
        let mutable lMin = System.Double.MaxValue
        let mutable lMax = System.Double.MinValue
        let mutable lPrev = System.Double.MinValue
        let mutable stop = false
        if print then printfn "Iters| Ep|Minib| Loss"
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
                elif i=iters-1 then
                    status <- sprintf "Iters=%d reached" iters
                    stop <- true
                elif lScalar < lMin then
                    lMin <- lScalar
                    status <- "🡾 New min"
                elif lScalar > lMax then
                    lMax <- lScalar
                    status <- "🡽 New max"
                elif lScalar < lPrev then
                    status <- "🡾"
                else
                    status <- "🡽"

                if print && ((i+1) % printEvery = 0 || i = 0 || stop) then
                    if stop then printEnd <- "\n"
                    printf "%s%4d | %d | %d/%d | %e %s%s%s" printPrefix (i+1) (epoch+1) (bi+1) dataloader.length lScalar status printPostfix printEnd
                lPrev <- lScalar
                not stop
            ) |> Seq.iter ignore
        
    static member sgd(f, x0:Tensor, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string, ?printNewLine:bool) =
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
        Optimizer.optimizeFun(update, x0, ?iters=iters, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix, ?printNewLine=printNewLine)

    static member sgd(model, dataloader, loss, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool, ?iters:int, ?threshold:double, ?print:bool, ?printEvery:int, ?printPrefix:string, ?printPostfix:string, ?printNewLine:bool) =
        let optimizer = SGD(model, ?lr=lr, ?momentum=momentum, ?nesterov=nesterov, ?weightDecay=weightDecay, ?reversible=reversible)
        Optimizer.optimizeModel(model, optimizer, dataloader, loss, ?iters=iters, ?threshold=threshold, ?print=print, ?printEvery=printEvery, ?printPrefix=printPrefix, ?printPostfix=printPostfix, ?printNewLine=printNewLine)



and SGD(model, ?lr:Tensor, ?momentum:Tensor, ?nesterov:bool, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let lr = defaultArg lr (dsharp.tensor(0.001))
    let nesterov = defaultArg nesterov true
    let reversible = defaultArg reversible false
    let mutable momBuffer = ParameterDict()
    let mutable momInit = false
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


and Adam(model, ?lr:Tensor, ?beta1:Tensor, ?beta2:Tensor, ?eps:Tensor, ?weightDecay:Tensor, ?reversible:bool) =
    inherit Optimizer(model)
    let lr = defaultArg lr (dsharp.tensor(1e-3))
    let beta1 = defaultArg beta1 (dsharp.tensor(0.9))
    let beta2 = defaultArg beta2 (dsharp.tensor(0.999))
    let eps = defaultArg eps (dsharp.tensor(1e-8))
    let reversible = defaultArg reversible false    
    let mutable stateStep = -1
    let mutable stateExpAvg = dsharp.zero()
    let mutable stateExpAvgSq = dsharp.zero()
    override o.updateRule name t =
        let mutable d = t.derivative
        match weightDecay with
        | Some wd -> d <- d.add(t.primal * wd)
        | None -> ()
        if stateStep = -1 then
            stateExpAvg <- t.zerosLike()

        stateStep <- stateStep + 1
        if reversible then
            t - lr * d
        else
            t.primal - lr * d