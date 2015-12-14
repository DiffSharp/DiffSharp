// The function that generates embedded Reber grammars.
// This is a more difficult kind with long term dependencies.

// http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/reberGrammar.php

type reberNode =
    | NodeF  // Only outputs B.
    | NodeS  // Can only receive B. Outputs T or P.

    | Node0a // Can only receive T. Outputs B.
    | Node1a // Can only receive B. Outputs T or P.
    | Node2a // Can receive T or S. Outputs S or X.
    | Node3a // Can receive P or T. Outputs V or T.
    | Node4a // Can receive X or P. Outputs X or S.
    | Node5a // Can only receive V. Outputs P or V.
    | Node6a // Can receive S or V. Outputs E.
    | Node7a // Can only receive E. Outputs T.

    | Node0b // Can only receive P. Outputs B.
    | Node1b // Can only receive B. Outputs T or P.
    | Node2b // Can receive T or S. Outputs S or X.
    | Node3b // Can receive P or T. Outputs V or T.
    | Node4b // Can receive X or P. Outputs X or S.
    | Node5b // Can only receive V. Outputs P or V.
    | Node6b // Can receive S or V. Outputs E.
    | Node7b // Can only receive E. Outputs P.

    | Node8  // Can receive T or P. Outputs E.

let rng = System.Random()

let b_string = [|1.0f;0.0f;0.0f;0.0f;0.0f;0.0f;0.0f|]
let t_string = [|0.0f;1.0f;0.0f;0.0f;0.0f;0.0f;0.0f|]
let p_string = [|0.0f;0.0f;1.0f;0.0f;0.0f;0.0f;0.0f|]
let s_string = [|0.0f;0.0f;0.0f;1.0f;0.0f;0.0f;0.0f|]
let x_string = [|0.0f;0.0f;0.0f;0.0f;1.0f;0.0f;0.0f|]
let v_string = [|0.0f;0.0f;0.0f;0.0f;0.0f;1.0f;0.0f|]
let e_string = [|0.0f;0.0f;0.0f;0.0f;0.0f;0.0f;1.0f|]

let t_p_string = [|0.0f;1.0f;1.0f;0.0f;0.0f;0.0f;0.0f|]
let t_v_string = [|0.0f;1.0f;0.0f;0.0f;0.0f;1.0f;0.0f|]
let s_x_string = [|0.0f;0.0f;0.0f;1.0f;1.0f;0.0f;0.0f|]
let p_v_string = [|0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f|]


let rec make_random_reber_string str list prediction node =
    match node with
        | NodeF ->
            make_random_reber_string "B" [b_string] [b_string] NodeS
        | NodeS ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") (t_string::list) (t_p_string::prediction) Node0a else make_random_reber_string (str+"P") (p_string::list) (t_p_string::prediction) Node0b

        | Node0a ->
            make_random_reber_string (str+"B") (b_string::list) (b_string::prediction) Node1a
        | Node1a ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") (t_string::list) (t_p_string::prediction) Node2a else make_random_reber_string (str+"P") (p_string::list) (t_p_string::prediction) Node3a
        | Node2a ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"S") (s_string::list) (s_x_string::prediction) Node2a else make_random_reber_string (str+"X") (x_string::list) (s_x_string::prediction) Node4a
        | Node3a ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") (t_string::list) (t_v_string::prediction) Node3a else make_random_reber_string (str+"V") (v_string::list) (t_v_string::prediction) Node5a
        | Node4a ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"X") (x_string::list) (s_x_string::prediction) Node3a else make_random_reber_string (str+"S") (s_string::list) (s_x_string::prediction) Node6a
        | Node5a ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"P") (p_string::list) (p_v_string::prediction) Node4a else make_random_reber_string (str+"V") (v_string::list) (p_v_string::prediction) Node6a
        | Node6a ->
            make_random_reber_string (str+"E") (e_string::list) (e_string::prediction) Node7a
        | Node7a ->
            make_random_reber_string (str+"T") (t_string::list) (t_string::prediction) Node8

        | Node0b ->
            make_random_reber_string (str+"B") (b_string::list) (b_string::prediction) Node1b
        | Node1b ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") (t_string::list) (t_p_string::prediction) Node2b else make_random_reber_string (str+"P") (p_string::list) (t_p_string::prediction) Node3b
        | Node2b ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"S") (s_string::list) (s_x_string::prediction) Node2b else make_random_reber_string (str+"X") (x_string::list) (s_x_string::prediction) Node4b
        | Node3b ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") (t_string::list) (t_v_string::prediction) Node3b else make_random_reber_string (str+"V") (v_string::list) (t_v_string::prediction) Node5b
        | Node4b ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"X") (x_string::list) (s_x_string::prediction) Node3b else make_random_reber_string (str+"S") (s_string::list) (s_x_string::prediction) Node6b
        | Node5b ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"P") (p_string::list) (p_v_string::prediction) Node4b else make_random_reber_string (str+"V") (v_string::list) (p_v_string::prediction) Node6b
        | Node6b ->
            make_random_reber_string (str+"E") (e_string::list) (e_string::prediction) Node7b
        | Node7b ->
            make_random_reber_string (str+"P") (p_string::list) (p_string::prediction) Node8

        | Node8 ->
            (str+"E"), ((e_string::list) |> List.rev), ((e_string::prediction) |> List.rev)

open System.Collections.Generic
let make_reber_set num_examples =
    let mutable c = 0
    let reber_set = new HashSet<string * float32 [] list * float32 [] list>()

    while c < num_examples do
        if reber_set.Add (make_random_reber_string "" [] [] NodeF) then c <- c+1
    reber_set
