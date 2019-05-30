open Base
open Torch

let sub = Var_store.sub
let subi = Var_store.subi
let relu6 xs = Tensor.(min (relu xs) (f 6.))

(* Conv2D + BatchNorm2D + ReLU6 *)
let cbr vs ksize ~stride ~input_dim ?(groups = 1) output_dim =
  let ksize, padding =
    match ksize with
    | `k1 -> 1, 0
    | `k3 -> 3, 1
  in
  let conv =
    Layer.conv2d_
      (subi vs 0)
      ~ksize
      ~stride
      ~padding
      ~groups
      ~use_bias:false
      ~input_dim
      output_dim
  in
  let bn = Layer.batch_norm2d (subi vs 1) output_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv xs |> Layer.apply_ bn ~is_training |> relu6)

(* Inverted residual block. *)
let inv vs ~stride ~expand_ratio ~input_dim output_dim =
  let vs = sub vs "conv" in
  let hidden_dim = input_dim * expand_ratio in
  let use_residual = input_dim = output_dim && stride = 1 in
  let cbr0, nid =
    if expand_ratio = 1
    then Layer.id_, 0
    else cbr (subi vs 0) `k1 ~stride:1 ~input_dim hidden_dim, 1
  in
  let cbr1 = cbr (subi vs nid) `k3 ~stride ~groups:hidden_dim ~input_dim hidden_dim in
  let conv =
    Layer.conv2d_
      (subi vs (nid + 1))
      ~ksize:1
      ~stride:1
      ~use_bias:false
      ~input_dim:hidden_dim
      output_dim
  in
  let bn = Layer.batch_norm2d (subi vs (nid + 2)) output_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply_ cbr0 xs ~is_training
      |> Layer.apply_ cbr1 ~is_training
      |> Layer.apply conv
      |> Layer.apply_ bn ~is_training
      |> fun ys -> if use_residual then Tensor.(xs + ys) else ys)

let blocks =
  (* t, c, n, s *)
  [ 1, 16, 1, 1
  ; 6, 24, 2, 2
  ; 6, 32, 3, 2
  ; 6, 64, 4, 2
  ; 6, 96, 3, 1
  ; 6, 160, 3, 2
  ; 6, 320, 1, 1
  ]

let v2 vs ~num_classes =
  let in_dim = 32 in
  let vs_f = sub vs "features" in
  let vs_c = sub vs "classifier" in
  let init_cbr = cbr (subi vs_f 0) `k3 ~stride:2 ~input_dim:3 in_dim in
  let layer_idx = ref 0 in
  let last_dim, layers =
    List.fold_map blocks ~init:in_dim ~f:(fun in_dim (t, c, nn, s) ->
        let layer =
          List.range 0 nn
          |> List.map ~f:(fun idx ->
                 Int.incr layer_idx;
                 let input_dim, stride = if idx = 0 then in_dim, s else c, 1 in
                 inv (subi vs_f !layer_idx) ~stride ~expand_ratio:t ~input_dim c)
          |> Layer.fold_
        in
        c, layer)
  in
  let layers = Layer.fold_ layers in
  Int.incr layer_idx;
  let final_cbr = cbr (subi vs_f !layer_idx) `k1 ~stride:1 ~input_dim:in_dim last_dim in
  let final_linear = Layer.linear (subi vs_c 1) ~input_dim:last_dim num_classes in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply_ init_cbr xs ~is_training
      |> Layer.apply_ layers ~is_training
      |> Layer.apply_ final_cbr ~is_training
      |> Tensor.dropout ~p:0.2 ~is_training
      |> Tensor.mean2 ~dim:[ 2 ] ~keepdim:false
      |> Tensor.mean2 ~dim:[ 2 ] ~keepdim:false
      |> Layer.apply final_linear)
