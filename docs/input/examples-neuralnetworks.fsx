(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"

(**

<div class="row">
    <div class="span9">
    <div class="well well-small" id="nuget" style="background-color:#E0EBEB">
        <b>Please note:</b> this is an introductory example and therefore the code is kept very simple. More advanced cases, including recurrent and convolutional networks, will be released as part of a separate library built on top of DiffSharp.
    </div>
    </div>
</div>

Neural Networks
===============

[Artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) are computational models inspired by biological nervous systems, capable of approximating functions depending on a large number of inputs. A network is defined by a connectivity structure and a set of weights between interconnected processing units ("neurons"). Optimizing, or tuning, the set of weights for a given task makes neural networks capable of learning.

Let's create a [feedforward neural network](http://en.wikipedia.org/wiki/Feedforward_neural_network) and use DiffSharp for implementing the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) algorithm for training it. As mentioned before, backpropagation is just a special case of reverse mode AD.

We start by defining our neural network structure.

*)

open DiffSharp.AD.Float64
open DiffSharp.Util

// A layer of neurons
type Layer =
    {mutable W:DM  // Weight matrix
     mutable b:DV  // Bias vector
     a:DV->DV}     // Activation function

// A feedforward network of neuron layers
type Network =
    {layers:Layer[]} // The layers forming this network

(** 

The network will consist of several layers of neurons. Each neuron works by taking inputs $\{x_1, \dots, x_n\}$ and calculating the activation (output)

$$$
  a = \sigma \left(\sum_{i} w_i x_i + b\right) \; ,

where $w_i$ are synapse weights associated with each input, $b$ is a bias, and $\sigma$ is an [activation function](http://en.wikipedia.org/wiki/Activation_function) representing the rate of [action potential](http://en.wikipedia.org/wiki/Action_potential) firing in the neuron.

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-neuralnetworks-neuron.png" alt="Chart" style="width:400px;"/>
    </div>
</div>

A conventional choice for the activation function had been the [sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function) $\sigma (z) = 1 / (1 + e^{-z})$ for a long period because of its simple derivative and gain control properties. Recently the hyperbolic tangent $\tanh$ and the [rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) $\textrm{ReLU}(z) = \max(0, z)$ have been more popular choices due to their convergence and training performance characteristics.

Now let's write the network evaluation code and a function for creating a given network configuration and initializing the weights and biases with small random values. In practice, proper weight initialization would depend on the network structure and the type of activation functions used, as it has been demonstrated to have an important effect on training convergence.

Network evaluation is implemented using linear algebra, where we have a weight matrix $\mathbf{W}^l$ holding the weights of all neurons in layer $l$. The elements of this matrix $w_{ij}$ represent the weight between the $j$-th neuron in layer $l - 1$ and the $i$-th neuron in layer $l$. We also have a bias vector $\mathbf{b}^l$ holding the biases of all neurons in layer $l$ (one bias per neuron). Thus evaluating the network is just a matter of computing the activation

$$$
  \mathbf{a}^l = \mathbf{W}^l \mathbf{a}^{l - 1} + \mathbf{b}^l \; ,

for each layer and passing the activation vector (output) of each layer as the input to the next layer, until we have the network output as the output vector of the last layer.
*)

let runLayer (x:DV) (l:Layer) =
    l.W * x + l.b |> l.a

let runNetwork (x:DV) (n:Network) =
    Array.fold runLayer x n.layers


let rnd = System.Random()

// Initialize a fully connected feedforward neural network
// Weights and biases between -0.5 and 0.5
// l : number of inputs and number of neurons in each subsequent layer
let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        {W = DM.init l.[i + 1] l.[i] (fun _ _ -> -0.5 + rnd.NextDouble())
         b = DV.init l.[i + 1] (fun _ -> -0.5 + rnd.NextDouble())
         a = sigmoid})}
(**

This gives us an easily scalable feedforward network architecture capable of expressing any number of inputs, outputs, and hidden layers. The network is fully connected, meaning that each neuron in a layer receives the outputs of all the neurons in the previous layer as its input.

For example,

*)

let net1 = createNetwork [|3; 4; 2|]

(**

would give us the following network with 3 input nodes, a hidden layer with 4 neurons, and an output layer with 2 neurons:

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-neuralnetworks-network.png" alt="Chart" style="width:400px;"/>
    </div>
</div>

We can also have more than one hidden layer.

For training networks, we will make use of reverse mode AD for propagating an error $E$ at the output backwards through the network weights. This will give us the partial derivative of the error at the output with respect to each weight $w_i$ and bias $b_i$ in the network, which we will then use in an update rule

$$$
 \begin{eqnarray*}
 \Delta w_i &=& -\eta \frac{\partial E}{\partial w_i} \; ,\\
 \Delta b_i &=& -\eta \frac{\partial E}{\partial b_i} \; ,\\
 \end{eqnarray*}

where $\eta$ is the learning rate.

We will use the [quadratic error](https://en.wikipedia.org/wiki/Mean_squared_error)

$$$
 E = \sum_i \Vert \mathbf{y}_i - \mathbf{a}(\mathbf{x}_i) \Vert^{2} \; ,

where $\mathbf{y}_i$ are the training targets and $\mathbf{a}(\mathbf{x}_i)$ are the vectors of outputs at the last layer of the network when training inputs $\mathbf{x}_i$ are supplied to the first layer.

Please see the [Nested AD](gettingstarted-nestedad.html) page for a better understanding of the low-level usage of adjoints and their backwards propagation.

*)

// The backpropagation algorithm
// n: network to be trained
// eta: learning rate
// epsilon: error threshold
// timeout: maximum number of iterations
// x: training input vectors
// y: training target vectors
let backprop (n:Network) eta epsilon timeout (x:DV[]) (y:DV[]) =
    let i = DiffSharp.Util.GlobalTagger.Next
    seq {for j in 0 .. timeout do
            for l in n.layers do
                l.W <- l.W |> makeReverse i
                l.b <- l.b |> makeReverse i

            let error = Array.map2 (fun x y -> DV.l2normSq (y - runNetwork x n)) x y |> Array.sum
            error |> reverseProp (D 1.) // Propagate adjoint value 1 backward

            for l in n.layers do
                l.W <- primal (l.W.P - eta * l.W.A)
                l.b <- primal (l.b.P - eta * l.b.A)

            printfn "Iteration %i, error %f" j (float error)
            if j = timeout then printfn "Failed to converge within %i steps." timeout
            yield float error}
    |> Seq.takeWhile ((<) epsilon)

(**

Using reverse mode AD here has two big advantages: (1) it makes the backpropagation code succinct and straightforward to write and maintain; and (2) it allows us to freely choose activation functions without the burden of coding their derivatives or modifying the backpropagation code accordingly.

We can now test the algorithm by training some networks.

It is known that [linearly separable](http://en.wikipedia.org/wiki/Linear_separability) rules such as [logical disjunction](http://en.wikipedia.org/wiki/Logical_disjunction) can be learned by a single neuron.

*)
open FSharp.Charting

let ORx = [|toDV [0.; 0.]
            toDV [0.; 1.]
            toDV [1.; 0.]
            toDV [1.; 1.]|]
let ORy = [|toDV [0.]
            toDV [1.]
            toDV [1.]
            toDV [1.]|]

// 2 inputs, one layer with one neuron
let net2 = createNetwork [|2; 1|]

// Train
let train2 = backprop net2 0.9 0.005 10000 ORx ORy

// Plot the error during training
Chart.Line train2

(**

    [lang=cs]
    val net2 : Network = {layers = [|{W = DM [[0.230677625; 0.1414874814]];
                                      b = DV [|0.4233988253|];}|];}

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Linearly inseparable problems such as [exclusive or](http://en.wikipedia.org/wiki/Exclusive_or) require one or more hidden layers to learn.
    
*)

let XORx = [|toDV [0.; 0.]
             toDV [0.; 1.]
             toDV [1.; 0.]
             toDV [1.; 1.]|]
let XORy = [|toDV [0.]
             toDV [1.]
             toDV [1.]
             toDV [0.]|]

// 2 inputs, 3 neurons in a hidden layer, 1 neuron in the output layer
let net3 = createNetwork [|2; 3; 1|]

// Train
let train3 = backprop net3 0.9 0.005 10000 XORx XORy

// Plot the error during training
Chart.Line train3

(**

    [lang=cs]
    val net3 : Network =
      {layers =
        [|{W = DM [[-0.04536837132; -0.3447727025]
                   [-0.07626016418; 0.06522091877]
                   [0.2581558948; 0.1597980939]];
           b = DV [|-0.3051199176; 0.2980325892; 0.4621827649|];};
          {W = DM [[-0.347911722; 0.2696812725; 0.2704776571]];
           b = DV [|-0.1477482923|];}|];}


<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Some Performance Tricks
-----------------------

For higher training performance, it is better to propagate matrices, instead of vectors, through the network. In other words, instead of treating the traning data as a set of input vectors $\mathbf{x}_i \in \mathbb{R}^n$ and target vectors $\mathbf{y}_i \in \mathbb{R}^m$, $i = 1 \dots d$, we can have an input matrix $\mathbf{X} \in \mathbb{R}^{n\times d}$ and a target matrix $\mathbf{Y} \in \mathbb{R}^{m\times d}$, where $d$ is the number of examples in the training set. In this scheme, vectors $\mathbf{x}_i$ form the columns of the input matrix $\mathbf{X}$, and propagating $\mathbf{X}$ through the layers computes the network's output for all $\mathbf{x}_i$ simultaneously. In DiffSharp, as in other linear algebra libraries, computing matrix-matrix multiplications are a lot more efficient than computing a series of matrix-vector multiplications.

Let's modify the network evaluation code to propagate matrices.
*)

// A layer of neurons
type Layer' =
    {mutable W:DM  // Weight matrix
     mutable b:DV  // Bias vector
     a:DM->DM}     // Activation function

// A feedforward network of neuron layers
type Network' =
    {layers:Layer'[]} // The layers forming this network

let runLayer' (x:DM) (l:Layer') =
    l.W * x + (DM.createCols x.Cols l.b) |> l.a

let runNetwork' (x:DM) (n:Network') =
    Array.fold runLayer' x n.layers

(**
The backpropagation code given previously computes the loss over the whole set of training cases at each iteration and uses simple gradient descent to iteratively decrease this loss. When the training set is large, this leads to training time bottlenecks. 

In practice, backpropagation is combined with stochastic gradient descent (SGD), which makes the duration of each training iteration independent from the training set size (also see the [SGD example](examples-stochasticgradientdescent.html)). Furthermore, instead of using one random case at a time to compute the loss, SGD is used with "minibatches" of more than one case (a small number compared to the full training set size). Minibatches allow us to exploit the matrix-matrix multiplication trick for performance and also have the added benefit of smoothing the SGD estimation of the true gradient.

Backpropagation combined with SGD and minibatches is the de facto standard for training neural networks.
*)

// Backpropagation with SGD and minibatches
// n: network
// eta: learning rate
// epsilon: error threshold
// timeout: maximum number of iterations
// mbsize: minibatch size
// loss: loss function
// x: training input matrix
// y: training target matrix
let backprop' (n:Network') (eta:float) epsilon timeout mbsize loss (x:DM) (y:DM) =
    let i = DiffSharp.Util.GlobalTagger.Next
    let mutable b = 0
    let batches = x.Cols / mbsize
    let mutable j = 0
    while j < timeout do
        b <- 0
        while b < batches do
            let mbX = x.[*, (b * mbsize)..((b + 1) * mbsize - 1)]
            let mbY = y.[*, (b * mbsize)..((b + 1) * mbsize - 1)]

            for l in n.layers do
                l.W <- l.W |> makeReverse i
                l.b <- l.b |> makeReverse i

            let error:D = loss (runNetwork' mbX n) mbY
            error |> reverseProp (D 1.)

            for l in n.layers do
                l.W <- primal (l.W.P - eta * l.W.A)
                l.b <- primal (l.b.P - eta * l.b.A)

            printfn "Epoch %i, minibatch %i, error %f" j b (float error)
            b <- b + 1
            if float error >= epsilon then j <- timeout
        j <- j + 1

(**
The Obligatory MNIST Example
----------------------------

The MNIST (for "Mixed National Institute of Standards and Technology") database of handwritten digits is


The following code reads the standard MNIST files into matrices.
*)

open System.IO

type MNIST =
    static member Load(filename, items) =
        let d = new BinaryReader(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read))
        let magicnumber = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            d.ReadBytes(min items maxitems)
            |> Array.map float |> DV
            |> DM.ofDV 1
        | 2051 -> // Images
            let maxitems = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let rows = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let cols = d.ReadInt32() |> System.Net.IPAddress.NetworkToHostOrder
            let n = min items maxitems
            d.ReadBytes(n * rows * cols)
            |> Array.map float |> DV
            |> DM.ofDV n
            |> DM.transpose
        | _ -> failwith "Given file is not in the MNIST format."
    static member Load(filename) = MNIST.Load(filename, System.Int32.MaxValue)

(**
Let's load 10,000 training images
*)

let mnistTrainX = MNIST.Load("C:/datasets/MNIST/train-images.idx3-ubyte", 10000)
let mnistTrainY = MNIST.Load("C:/datasets/MNIST/train-labels.idx1-ubyte", 10000)

(**
Loss
*)

let cross (x:DM) (y:DM) =
    (x |> DM.toCols |> Seq.mapi (fun i v -> (logsumexp v) - v.[int (float y.[0, i])]) |> Seq.sum) / x.Cols

(**
Create a network
*)

let l0 = {W = DM.init 300 785 (fun _ _ -> -0.075 + 0.15 * rnd.NextDouble())
          b = DV.init 300 (fun _ -> 0.)
          a = tanh}

let l1 = {W = DM.init 10 300 (fun _ _ -> -0.075 + 0.15 * rnd.NextDouble())
          b = DV.init 10 (fun _ -> 0.)
          a = id}

let nn = {layers = [|l0; l1|]}

(**
Train
*)

//backprop' nn 0.1 0.005 1000 1000 cross mnistTrainX mnistTrainY

(**
Test
*)

let mnistTestX = MNIST.Load("C:/datasets/MNIST/t10k-images.idx3-ubyte", 5)
let mnistTestY = MNIST.Load("C:/datasets/MNIST/t10k-labels.idx1-ubyte", 5)

let testY = runNetwork' mnistTestX nn |> primal
let testError = cross testY mnistTestY
let testPredict = testY |> DM.toCols |> Seq.map DV.maxIndex |> Seq.toArray

for i = 0 to testY.Cols - 1 do
    printfn "Predicted label: %i" testPredict.[i]
    printfn "Image:\n %s" ((mnistTestX.[*,i] |> DM.ofDV 28).Visualize())

(**
    [lang=cs]
    Predicted label: 7
    Image:
     DM : 28 x 28
                            
                            
                            
                            
                            
                            
                            
          ■■■■■■                
          ■■■■■■■■■■■■■■■■      
          ■■■■■■■■■■■■■■■■      
                ■ ■■■■ ■■■      
                      ■■■       
                      ■■■       
                     ■■■■       
                    ■■■■        
                    ■■■         
                    ■■■         
                   ■■■          
                  ■■■■          
                  ■■■           
                 ■■■■           
                ■■■■            
               ■■■■             
               ■■■■             
              ■■■■■             
              ■■■■■             
              ■■■               
                            

    Predicted label: 2
    Image:
     DM : 28 x 28
                            
                            
                            
              ■■■■■■■           
             ■■■■■■■■           
            ■■■■■■■■■■          
           ■■■■    ■■■          
           ■■■    ■■■■          
                  ■■■■          
                 ■■■■           
                ■■■■■           
                ■■■■            
               ■■■■             
               ■■■              
              ■■■■              
             ■■■■               
             ■■■■               
            ■■■■                
            ■■■                 
            ■■■■         ■■■■■  
            ■■■■■■■■■■■■■■■■■■■ 
            ■■■■■■■■■■■■■■■■■■■ 
             ■■■■■■■■■■■■       
                            
                            
                            
                            
                            

    Predicted label: 1
    Image:
     DM : 28 x 28
                            
                            
                            
                            
                    ■■■         
                    ■■■         
                    ■■          
                   ■■■          
                   ■■■          
                   ■■           
                  ■■■           
                  ■■■           
                  ■■■           
                 ■■■            
                 ■■■            
                 ■■■            
                ■■■             
                ■■■             
                ■■■             
                ■■■             
               ■■■■             
               ■■■              
               ■■■              
               ■■               
                            
                            
                            
                            

    Predicted label: 0
    Image:
     DM : 28 x 28
                            
                            
                            
                            
                 ■■■            
                 ■■■■           
                ■■■■■           
              ■■■■■■■■■         
              ■■■■■■■■■■        
             ■■■■■■■■■■■        
            ■■■■■■■■■■■■■       
            ■■■■■■    ■■■■      
            ■■■■      ■■■■      
            ■■■        ■■■      
            ■■         ■■■■     
           ■■■        ■■■■      
           ■■■       ■■■■■      
           ■■■      ■■■■■■      
           ■■■     ■■■■■■       
           ■■■■■■■■■■■■■        
           ■■■■■■■■■■■■■        
            ■■■■■■■■■■          
            ■■■■■■■■■           
              ■■■■■■            
                            
                            
                            
                            

    Predicted label: 4
    Image:
     DM : 28 x 28
                            
                            
                            
                            
                            
              ■■       ■■       
              ■■       ■■       
              ■■       ■■       
             ■■■       ■■       
            ■■■        ■■       
            ■■■        ■■       
           ■■■        ■■■       
           ■■■       ■■■■       
           ■■        ■■■        
           ■■        ■■■        
           ■■        ■■■        
           ■■■■■■■■■■■■■        
           ■■■■■■■■■■■■         
             ■■■■■  ■■■         
                    ■■■■        
                    ■■■         
                    ■■■         
                    ■■■         
                    ■■■         
                    ■■          
                            
                            
                        
*)