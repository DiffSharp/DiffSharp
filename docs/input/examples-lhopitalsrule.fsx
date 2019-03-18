(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**

l'Hôpital's Rule
================

[l'Hôpital's rule](https://en.wikipedia.org/wiki/L'H%C3%B4pital's_rule) is a method for evaluating [limits](https://en.wikipedia.org/wiki/Limit_of_a_function) involving [indeterminate forms](https://en.wikipedia.org/wiki/Indeterminate_form). The rule states that, under some conditions, the indeterminate limits

$$$
 \begin{eqnarray*}
 \lim_{x \to c} \frac{f(x)}{g(x)} &=& \frac{0}{0} \; ,\\
 \lim_{x \to c} \frac{f(x)}{g(x)} &=& \frac{\pm\infty}{\pm\infty}
 \end{eqnarray*}

can be found by differentiating the numerator and the denominator and taking the limit, that is

$$$
 \lim_{x \to c} \frac{f(x)}{g(x)} = \lim_{x \to c} \frac{f'(x)}{g'(x)} \; .

Let us try to find

$$$
 \lim_{x \to 0} \frac{2 \sin x - \sin 2x}{x - \sin x}

which involves the indeterminate form $0 / 0$.

*)

// Define f(x)
let f x = 2. * sin x - sin (2. * x)

// Define g(x)
let g x = x - sin x

// Try to evaluate the limit at x = 0
let lim = f 0. / g 0.

(**
As expected, we get a [nan](https://msdn.microsoft.com/en-us/library/system.double.nan.aspx) as a result, meaning the result of this operations is undefined.
*)

(*** hide, define-output: o1 ***)
printf "val lim : float = nan"
(*** include-output: o1 ***)

(**
 
Using DiffSharp, we can generate a sequence of repeated applications of l'Hôpital's rule.
*)

open DiffSharp.AD.Float64

// Differentiate f(x) and g(x) n times and evaluate the division
let lhopital f g n x = diffn n f x / diffn n g x

// Generate an infinite sequence of lhopital applications,
// starting from the undifferentiated limit (n = 0)
let lhseq f g x = Seq.initInfinite (fun n -> lhopital f g n x)

// Use lhseq with f(x) and g(x), at x = 0
let l = lhseq (fun x -> 2. * sin x - sin (2. * x)) (fun x -> x - sin x) (D 0.)

(**
The first four elements ($n = 0,\dots,4$) of this infinite sequence are   
*)

(*** hide, define-output: o2 ***)
printf "val l : seq [nan; nan; nan; D 6.0; ...]"
(*** include-output: o2 ***)

(**
For $n = 0$, we have the original indeterminate value of the limit (since the 0-th derivative of a function is itself). 

For $n = 3$, we have the value of this limit as

$$$
 \lim_{x \to 0} \frac{2 \sin x - \sin 2x}{x - \sin x} = 6 \; ,

after 3 applications of l'Hôpital's rule. 

We can check this by manually differentiating:

$$$
 \begin{eqnarray*}
 \lim_{x \to 0} \frac{2 \sin x - \sin 2x}{x - \sin x} &=& \lim_{x \to 0} \frac{2\cos x - 2\cos 2x}{1 - \cos x}\\
 &=& \lim_{x \to 0} \frac{-2 \sin x + 4 \sin 2x}{\sin x}\\
 &=& \lim_{x \to 0} \frac{-2 \cos x + 8 \cos 2x}{\cos x}\\
 &=& \frac{-2 + 8}{1}\\
 &=& 6 \; .
 \end{eqnarray*}

We can go further and automate this process by searching for the first element of the sequence that is not indeterminate.

*)

// Check if x is not indeterminate
let isdeterminate x =
    not (System.Double.IsInfinity(x)
    || System.Double.IsNegativeInfinity(x)
    || System.Double.IsPositiveInfinity(x)
    || System.Double.IsNaN(x))

// Find the limit of f(x) / g(x) at a given point
let findlim f g x = Seq.find (float >> isdeterminate) (lhseq f g x)

// Find the limit of f(x) / g(x) at x = 0
let lim2 = findlim (fun x -> 2. * sin x - sin (2. * x)) (fun x -> x - sin x) (D 0.)

(*** hide, define-output: o3 ***)
printf "val lim2 : D = D -6.0"
(*** include-output: o3 ***)

(**
Let us use our function to evaluate some other limits.

The limit

$$$
 \lim_{x \to 0} \frac{\sin \pi x}{\pi x}

has indeterminate form $0 / 0$ and can be evaluated by

$$$
 \lim_{x \to 0} \frac{\sin \pi x}{\pi x} = \lim_{y \to 0} \frac{\sin y}{y} = \lim_{y \to 0} \frac{\cos y}{1} = 1 \; .
*)

let lim3 = findlim (fun x -> sin (System.Math.PI * x)) (fun x -> System.Math.PI * x) (D 0.)

(*** hide, define-output: o4 ***)
printf "val lim3 : D = D 1.0"
(*** include-output: o4 ***)

(**
The limit

$$$
 \lim_{x \to 0} \frac{e^x - 1 - x}{x^2}

has indeterminate form $0 / 0$ and can be evaluated by

$$$
 \lim_{x \to 0} \frac{e^x - 1 - x}{x^2} = \lim_{x \to 0} \frac{e^x - 1}{2x} = \lim_{x \to 0} \frac{e^x}{2} = \frac{1}{2} \; .
*)

let lim4 = findlim (fun x -> exp x - 1 - x) (fun x -> x * x) (D 0.)

(*** hide, define-output: o5 ***)
printf "val lim4 : D = D 0.5"
(*** include-output: o5 ***)

(**

It should be noted that there are cases where repeated applications of l'Hôpital's rule do not lead to an answer without a transformation of variables.

*)