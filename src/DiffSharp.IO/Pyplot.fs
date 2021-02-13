namespace DiffSharp.IO

open DiffSharp
open Python.Runtime

type Pyplot() =
    let _gil = Py.GIL()
    let scope = Py.CreateScope()
    let exec(code) = scope.Exec(code)
    let _prep = exec("import matplotlib.pyplot as plt")
    // let pyList (l:float32[]) =
    //     let sb = System.Text.StringBuilder()
    //     let mutable prefix = "["
    //     for v in l do
    //         sb.Append(prefix) |> ignore
    //         sb.Append(v) |> ignore
    //         prefix <- ", "
    //     sb.Append("]") |> ignore
    //     sb.ToString()
    member _.plot(y:Tensor, ?label:string) =
        let label = defaultArg label ""
        // let y = y |> Seq.map float32 |> Seq.toArray |> pyList
        let y = y.ToString()
        exec(sprintf "plt.plot(%s, label='%s')" y label)
    member _.savefig(fileName) = exec(sprintf "plt.savefig('%s')" fileName)
    member _.show() = exec("plt.show()")
    member _.legend() = exec("plt.legend()")
    member _.tight_layout() = exec("plt.tight_layout()")