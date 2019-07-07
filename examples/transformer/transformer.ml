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

let sublayer_connection vs sublayer dim ~dropout =
  let layer_norm = layer_norm vs dim in
  Layer.of_fn_ (fun xs ~is_training ->
      let ys =
        Layer.forward layer_norm xs
        |> Layer.forward sublayer
        |> Tensor.dropout ~is_training ~p:dropout
      in
      Tensor.(xs + ys))

let attention ~query ~key ~value ~mask ~dropout ~is_training =
  let sqrt_d_k = Tensor.size query |> List.last_exn |> Float.of_int |> Float.sqrt in
  let scores = Tensor.matmul query (Tensor.transpose key ~dim0:(-2) ~dim1:(-1)) in
  let p_attn =
    let mask = Tensor.eq_scalar mask (Scalar.int 0) in
    Tensor.(scores / f sqrt_d_k)
    |> Tensor.masked_fill ~mask ~value:(Scalar.float (-1e9))
    |> Tensor.softmax ~dim:(-1)
    |> Tensor.dropout ~p:dropout ~is_training
  in
  Tensor.matmul p_attn value, p_attn

let () = ignore (sublayer_connection, attention)
