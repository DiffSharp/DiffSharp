

#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.1"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.Util

let t = dsharp.tensor([1,2,3]).argmax(0)
//let t = dsharp.tensor([[1,2,3]]).argmax(0)
//let t = dsharp.tensor([[1,2,3]]).argmax(1)
//let t = dsharp.tensor([[3,2,1]]).argmax(1)
//let t = dsharp.tensor([[3,3,3]]).argmax(1)
//let t = dsharp.tensor([[1,2,3],[4,5,6]], backend=Backend.Reference).argmax(1)
//let t = dsharp.tensor([[1,2,3],[4,5,6]], backend=Backend.Torch).argmax(0)
//let t = dsharp.tensor([[1,2,3],[4,5,6]]).argmax(2)
let t = dsharp.tensor([[1.;4.];[2.;3.]]).argmax(0)
let t1 = dsharp.tensor([4.;1.;20.;3.])
let t2 = dsharp.tensor([[1.;4.];[2.;3.]])
let t3 = dsharp.tensor([[[ 7.6884; 65.9125;  4.0114];
                     [46.7944; 61.5331; 40.1627];
                     [48.3240;  4.9910; 50.1571]];

                    [[13.4777; 65.7656; 36.8161];
                     [47.8268; 42.2229;  5.6115];
                     [43.4779; 77.8675; 95.7660]];

                    [[59.8422; 47.1146; 36.7614];
                     [71.6328; 18.5912; 27.7328];
                     [49.9120; 60.3023; 53.0838]]])
t2.argmin(0, keepDim=true)
t1.argmin(0)
t2.argmax(2, keepDim=)
