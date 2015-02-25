(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#r "../../packages/FSharp.Data.2.1.1/lib/net40/FSharp.Data.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
K-Means Clustering
==================

[K-means clustering](http://en.wikipedia.org/wiki/K-means_clustering) is a popular method in [cluster analysis](http://en.wikipedia.org/wiki/Cluster_analysis) for partitioning a given set of observations into $k$ clusters, where the observations in the same cluster are more similar to each other than to those in other clusters.

Given $d$ observations $\{\mathbf{x}_1,\dots,\mathbf{x}_d\}$, the observations are assigned to $k$ clusters $\mathbf{S} = \{S_1,\dots,S_k\}$ so as to minimize

$$$
  \underset{\mathbf{S}}{\textrm{argmin}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \left|\left| \mathbf{x} - \mathbf{\mu}_i \right|\right|^2 \; ,

where $\mathbf{\mu}_i$ is the mean of the observations in $S_i$.

The classical way of finding k-means partitionings is using a heuristic algorithm cycling through an _assignment step_, where observations are assigned to the cluster of the mean that they are currently closest to, and an _update step_ where the means are updated as the centroids of the observations that are currently assigned to them, until assignments no longer change.

Let us use an alternative approach and implement k-means clustering using the stochastic gradient descent algorithm that we introduced in [another example](examples-stochasticgradientdescent.html). This variety of k-means clustering has been proposed in the literature for addressing large-scale learning tasks, due to its superior performance.

We start with the generic stochastic gradient descent code, introduced in the [stochastic gradient descent example](examples-stochasticgradientdescent.html), that can be used for finding weights $\mathbf{w}$ optimizing a model function $f_{\mathbf{w}}: \mathbb{R}^n \to \mathbb{R}^m$ trained using a set of inputs $\mathbf{x}_i \in \mathbb{R}^n$ and outputs $\mathbf{y}_i \in \mathbb{R}^m$.

*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra

let rnd = new System.Random()

// Stochastic gradient descent
// f: function, w0: starting weights, eta: step size, epsilon: threshold, t: training set
let sgd f w0 (eta:float) epsilon (t:(Vector<float>*Vector<float>)[]) =
    let ta = Array.map (fun (x, y) -> Vector.map adj x, Vector.map adj y) t
    let rec desc w =
        let x, y = ta.[rnd.Next(ta.Length)]
        let g = grad (fun wi -> Vector.normSq (y - (f wi x))) w
        if Vector.normSq g < epsilon then w else desc (w - eta * g)
    desc w0


(**

The following code implements the k-means model in the form

$$$
  f_{\mathbf{w}}(\mathbf{x}) = \left|\left| \mathbf{x} - \mathbf{\mu}_{\ast} \right|\right|^2 \; ,

where $\mathbf{x} \in \mathbb{R}^n$,

$$$
  \mathbf{\mu}_{\ast} = \{ \mathbf{\mu}_i : \left|\left| \mathbf{x} - \mathbf{\mu}_i \right|\right|^2 \le \left|\left| \mathbf{x} - \mathbf{\mu}_j \right|\right|^2 \; , \; \textrm{for all} \; 1 \le j \le k \}

is the closest of the current means to the given point $\mathbf{X}$ and the current means $\mathbf{\mu}_i \in \mathbb{R}^n$ encoded in the weight vector $\mathbf{w} \in \mathbb{R}^{k\,n}$ are obtained by splitting it into subvectors $\{\mathbf{\mu}_1,\dots,\mathbf{\mu}_k\}$

$$$
  \mathbf{w} = \left[ \mathbf{\mu}_1 \, \mathbf{\mu}_2 \, \dots \, \mathbf{\mu}_k \right] \; .

A given set of $d$ observations are then supplied to the stochastic gradient descent algorithm as the training set where the inputs are $\{\mathbf{x}_1,\dots,\mathbf{x}_d\}$ and the expected output is always $0$.

An important thing to note here is that the **DiffSharp.AD.Reverse** module takes the derivative (via reverse mode AD) of this whole algorithm, which includes subprocedures, control flow, and random sampling, and makes the gradient calculations transparent. We do not need to concern ourselves with formulating the model in a closed-form expression for being able to define and then compute its derivative.

*)

let kmeans k eta epsilon (data:Vector<float>[]) =
    // (index of, squared distance to) the nearest mean to x
    let inline nearestm (x:Vector<_>) (means:Vector<_>[]) =
        means |> Array.mapi (fun i m -> i, Vector.normSq (x - m)) |> Array.minBy snd
    // Extract means from w
    let inline extractm (w:Vector<_>) =
        Array.init k (fun i -> Vector.init (w.Length/k) (fun j -> w.[i*(w.Length/k)+j]))
    // Squared distance of x to the nearest of the means encoded in w
    let inline dist (w:Vector<_>) (x:Vector<_>) = w |> extractm |> nearestm x |> snd
    // The training set
    let train = Array.zip data (Array.create data.Length (vector [0.]))
    let w0 = Array.init k (fun _ -> Vector.toArray data.[rnd.Next(data.Length)]) 
                |> Array.concat |> vector
    let wopt = sgd dist w0 eta epsilon train
    let means = extractm wopt
    let assign = Array.map (fun d -> (nearestm d means |> fst, d)) data
    Array.init k (fun i -> assign |> Array.filter (fun (j, d) -> i = j) |> Array.map snd)

(**

Now we can test the algorithm in a two-dimensional space, using a set of randomly generated points.

*)

// Generate 200 random points
let data = Array.init 200 (fun _ -> (Vector.init 2 (fun _ -> rnd.NextDouble())))

// Partition the data into 5 clusters
let clusters = kmeans 5 0.01 0.01 data

(**
    
*)

open FSharp.Charting

let plotClusters (c:Vector<float>[][]) =
    Chart.Combine(
        List.init c.Length (fun i -> 
            Chart.Point(Array.map (fun (d:Vector<_>) -> d.[0], d.[1]) c.[i], MarkerSize = 10))
        )

plotClusters clusters

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kmeansclustering-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>
 
Compared to the commonly used batch-update k-means algorithm running through all the observations at each step, this algorithm has the advantage of running independent from the number of observations in the data set.

*)

// Generate 10000 random points
let data2 = Array.init 10000 (fun _ -> (Vector.init 2 (fun _ -> rnd.NextDouble())))

// Partition the data into 8 clusters
let clusters2 = kmeans 8 0.01 0.01 data2

plotClusters clusters2

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kmeansclustering-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Finally, we can test our algorithm with the [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set) that is commonly used for demonstrations. The data set contains four morphological features (_sepal length_, _sepal width_, _petal length_, _petal width_) of Iris flowers belonging to three related species. A version of the data set can be found [here](https://dataminingproject.googlecode.com/svn-history/r44/DataMiningApp/datasets/Iris/iris.csv).

*)

open FSharp.Data

let iris = new CsvProvider<"./resources/iris.csv">()

let irisData = 
    iris.Rows
    |> Seq.map (fun r -> vector [float r.``Sepal Width``; float r.``Petal Length``])
    |> Seq.toArray

let irisClusters = kmeans 3 0.01 0.01 irisData

plotClusters irisClusters

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kmeansclustering-chart3.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Our clustering of the _sepal width_ - _petal length_ data correctly predicts the actual assignment of these features to the three flower species, which can be seen below ([image](http://en.wikipedia.org/wiki/Iris_flower_data_set#mediaviewer/File:Anderson%27s_Iris_data_set.png) by user [Indon](http://commons.wikimedia.org/wiki/User:Indon) on Wikimedia Commons, CC BY-SA 3.0).

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kmeansclustering-chart4.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)