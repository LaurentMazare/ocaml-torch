(* Transformer Network
   As introduced in "Attention Is All You Need", Vaswani et al. 2017.

   http://nlp.seas.harvard.edu/2018/04/03/attention.html
*)
open Base
open Torch

let layer_norm vs dim =
  let a = Var_store.new_var vs ~name:"a" ~shape:[ dim ] ~init:Ones in
  let b = Var_store.new_var vs ~name:"b" ~shape:[ dim ] ~init:Zeros in
  Layer.of_fn (fun xs ->
      let mu = Tensor.mean2 xs ~dim:[ -1 ] ~keepdim:true in
      let sigma = Tensor.std1 xs ~dim:[ -1 ] ~unbiased:true ~keepdim:true in
      Tensor.((a * (xs - mu) / (sigma + f 1e-6)) + b))

let () = ignore (layer_norm, ())
