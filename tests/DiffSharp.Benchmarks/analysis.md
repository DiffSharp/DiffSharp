

This project is for benchmarking, do as you want with it.



### Analysis 1 (8/9/2020)

I took some measurements of repeated "tensor + tensor" and approximately fitted them to a cost model based on 

* N = overall number of primitive additions
* n = number of elements in the tensor
* t = c1 * N + c2 / (N/n)
* c1 = u/s per primitive addition
* c2 = u/s per overall tensor addition operation (not including creating tensors nor getting data to the GPU)

Very approximate figures

* Torch CPU, best c1 = 0.0010, c2 = approx 8.0

* Torch GPU, best c1 = 0.000056, c2 = approx 75.0

* Reference CPU, best c1 = 0.0025, c2 = approx 1.0 

These are pretty much as you'd expect 

* setting up operations on the GPU is expensive (c2 = 75.0) then very fast (c1 = 0.000056)

* setting up Torch operations on the CPU is non-zero cost (c2 = 8.0) then fast (c1 = 0.0010)

* setting up Reference implementation operations on the CPU is low cost (c2) then slow (c1 = 0.0025)

* the reference backend has low overhead to reach down into the actual tensor data but is significantly slower on actual floating point performance 

* overall we don't expect DiffSharp to be fast on tiny tensors.  Indeed for this particular operation it's not until you have about tensors of about size 10,000 that the Torch CPU or GPU backends become faster than the current reference backend (note however that in the reference backend the addition operation is implemented in a tight loop and .NET will do an OK job on this - many other operations in the reference backend are implemented far less efficiently)


I also separately checked the cost of adding dummy fields to "RawTensor" - each dummy copy of "shape" seemed to add about 3% cost to Torch CPU for tiny tensors,
this obviously becomes less important as tensors grow in size.




