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
  Tensor.matmul p_attn value

let multi_headed_attention vs dim ~heads ~dropout =
  let linear_q = Layer.linear vs ~input_dim:dim dim in
  let linear_k = Layer.linear vs ~input_dim:dim dim in
  let linear_v = Layer.linear vs ~input_dim:dim dim in
  let linear_out = Layer.linear vs ~input_dim:dim dim in
  let d_k = dim / heads in
  fun ~query ~key ~value ~mask ~is_training ->
    let mask = Tensor.unsqueeze mask ~dim:1 in
    let batch_size = Tensor.size query |> List.hd_exn in
    let view_transpose xs =
      Tensor.view xs ~size:[ batch_size; -1; heads; d_k ]
      |> Tensor.transpose ~dim0:1 ~dim1:2
    in
    let query = Layer.forward linear_q query |> view_transpose in
    let key = Layer.forward linear_k key |> view_transpose in
    let value = Layer.forward linear_v value |> view_transpose in
    attention ~query ~key ~value ~mask ~dropout ~is_training
    |> Tensor.transpose ~dim0:1 ~dim1:2
    |> Tensor.contiguous
    |> Tensor.view ~size:[ batch_size; -1; heads * d_k ]
    |> Layer.forward linear_out

let () = ignore (sublayer_connection, multi_headed_attention)
