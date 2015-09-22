#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"


open DiffSharp.AD.Float64

// Leapfrog integrator
// u: potential energy function
// k: kinetic energy function
// d: integration step size
// steps: number of integration steps
// (x0, p0): initial position and momentum vectors
let leapFrog (u:DV->D) (k:DV->D) (d:D) steps (x0, p0) =
    let hd = d / 2.
    [1..steps] 
    |> List.fold (fun (x, p) _ ->
        let p' = p - hd * grad u x
        let x' = x + d * grad k p'
        x', p' - hd * grad u x') (x0, p0)


let Rnd = new System.Random()

// Uniform random number ~U(0, 1)
let rnd() = Rnd.NextDouble()

// Standard normal random number ~N(0, 1), via Box-Muller transform
let rec rndn() =
    let x, y = rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0
    let s = x * x + y *y
    if s > 1.0 then rndn() else x * sqrt (-2.0 * (log s) / s)


// Hamiltonian Monte Carlo
// n: number of samples wanted
// hdelta: step size for Hamiltonian dynamics
// hsteps: number of steps for Hamiltonian dynamics
// x0: initial state
// f: target distribution function
let hmc n hdelta hsteps (x0:DV) (f:DV->D) =
    let u x = -log (f x) // potential energy
    let k p = (p * p) / D 2. // kinetic energy
    let hamilton x p = u x + k p
    let x = ref x0
    [|for i in 1..n do
        let p = DV.init x0.Length (fun _ -> rndn())
        let x', p' = leapFrog u k hdelta hsteps (!x, p)
        if rnd() < float (exp ((hamilton !x p) - (hamilton x' p'))) then x := x'
        yield !x|]


// Multivariate normal distribution (any dimension)
// mu: mean vector
// sigma: covariance matrix
let multiNormal (mu:DV) (sigma:DM) =
    fun (x:DV) ->
        let s = sigma |> DM.inverse
        exp (-((x - mu) * s * (x - mu)) / D 2.)

//
//// Take 10000 samples from a bivariate normal distribution
//// mu1 = 0, mu2 = 0, correlation = 0.8
let samples = 
    multiNormal (toDV [D 0.; D 0.]) (toDM [[D 1.; D 0.8]; [D 0.8; D 1.]])
    |> hmc 10000 (D 0.1) 10 (toDV [D 0.; D 0.])


open FSharp.Charting

Chart.Point(samples |> Array.map (fun v -> float v.[0], float v.[1]), MarkerSize = 2)


