open Base
open Torch

let relu6 xs = Tensor.(max (relu xs) (f 6.))

let conv_bn vs ~n ~ksize ~stride ~input_dim output_dim =
  let ksize, padding =
    match ksize with
    | `k1 -> 1, 0
    | `k3 -> 3, 1
  in
  let conv =
    Layer.conv2d_ vs
      ~n:(n 0) ~ksize ~stride ~padding ~use_bias:false ~input_dim output_dim
  in
  let bn = Layer.batch_norm2d vs ~n:(n 1) output_dim in
  Layer.of_fn_ (fun xs ~is_training ->
    Layer.apply conv xs
    |> Layer.apply_ bn ~is_training)

let inverted_residual vs ~n ~stride ~expand_ratio ~input_dim output_dim =
  let n ~base i = N.(n / Int.to_string (base + i)) in
  let hidden_dim = input_dim * expand_ratio in
  let use_residual = input_dim = output_dim && stride = 1 in
  let conv0 =
    if expand_ratio = 1
    then Layer.id_
    else
      Layer.fold_
        [ conv_bn vs ~n:(n ~base:0) ~ksize:`k1 ~stride:1 ~input_dim hidden_dim
        ; Layer.of_fn_ (fun xs ~is_training:_ -> relu6 xs)
        ]
  in
  let base = if expand_ratio = 1 then 0 else 3 in
  let conv1 = conv_bn vs ~n:(n ~base) ~ksize:`k3 ~stride ~input_dim hidden_dim in
  let conv2 =
    conv_bn vs ~n:(n ~base:(base + 3)) ~ksize:`k1 ~stride:1 ~input_dim:hidden_dim output_dim
  in
  Layer.of_fn_(fun xs ~is_training ->
    Layer.apply_ conv0 xs ~is_training
    |> Layer.apply_ conv1 ~is_training
    |> relu6
    |> Layer.apply_ conv2 ~is_training
    |> fun ys -> if use_residual then Tensor.(xs + ys) else ys)

let v2 vs ~num_classes =
  let input_dim = 32 in
  let n_features = N.(root / "features") in
  let n d i = N.(n_features / Int.to_string d / Int.to_string i) in
  let initial_conv = conv_bn vs ~n:(n 0) ~ksize:`k3 ~stride:2 ~input_dim:3 input_dim in
  let last_dim, layers =
    let layer_idx = ref 0 in
    (* t, c, n, s *)
    [ 1, 16, 1, 1
    ; 6, 24, 2, 2
    ; 6, 32, 3, 2
    ; 6, 64, 4, 2
    ; 6, 96, 3, 1
    ; 6, 160, 3, 2
    ; 6, 320, 1, 1
    ]
    |> List.fold_map ~init:input_dim ~f:(fun input_dim (t, c, nn, s) ->
      let layer =
        List.init nn ~f:(fun idx ->
          Int.incr layer_idx;
          let n = N.(n_features / Int.to_string !layer_idx / "conv") in
          let input_dim, stride = if idx = 0 then input_dim, s else c, 1 in
          inverted_residual vs ~n ~stride ~expand_ratio:t ~input_dim c)
        |> Layer.fold_
      in
      c, layer)
  in
  let layers = Layer.fold_ layers in
  let final_conv = conv_bn vs ~n:(n 18) ~ksize:`k1 ~stride:1 ~input_dim last_dim in
  let final_linear =
    Layer.linear vs ~n:N.(root / "classifier" / "1") ~input_dim:last_dim num_classes
  in
  Layer.of_fn_ (fun xs ~is_training ->
    let batch_size = Tensor.shape xs |> List.hd_exn in
    Layer.apply_ initial_conv xs ~is_training
    |> relu6
    |> Layer.apply_ layers ~is_training
    |> Layer.apply_ final_conv ~is_training
    |> relu6
    |> Tensor.dropout ~p:0.2 ~is_training
    |> Tensor.view ~size:[ batch_size; last_dim ]
    |> Layer.apply final_linear)
