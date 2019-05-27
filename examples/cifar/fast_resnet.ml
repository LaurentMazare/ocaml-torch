(* Fast Resnet variant
   Adapted from https://github.com/davidcpage/cifar10-fast/
*)
open Base
open Torch

let batch_size = 512
let epochs = 24

let lr_schedule =
  let schedule = Optimizer.Linear_interpolation.create [ 0., 0.; 5., 0.4; 24., 0. ] in
  fun ~batch_idx ~batches_per_epoch ~epoch_idx ->
    let epoch_idx =
      Float.of_int (epoch_idx - 1)
      +. (Float.of_int batch_idx /. Float.of_int batches_per_epoch)
    in
    Optimizer.Linear_interpolation.eval schedule epoch_idx

let conv_bn vs ~c_in ~c_out =
  let open Layer in
  fold_
    [ conv2d_ vs ~ksize:3 ~stride:1 ~padding:1 ~use_bias:false ~input_dim:c_in c_out
      |> with_training
    ; batch_norm2d vs c_out ~w_init:Ones
    ; of_fn_ (fun xs ~is_training:_ -> Tensor.relu_ xs)
    ]

let layer vs ~c_in ~c_out =
  let pre = conv_bn vs ~c_in ~c_out in
  let block1 = conv_bn vs ~c_in:c_out ~c_out in
  let block2 = conv_bn vs ~c_in:c_out ~c_out in
  Layer.of_fn_ (fun xs ~is_training ->
      let pre = Layer.apply_ pre xs ~is_training |> Tensor.max_pool2d ~ksize:(2, 2) in
      Layer.apply_ block1 pre ~is_training
      |> Layer.apply_ block2 ~is_training
      |> fun ys -> Tensor.( + ) pre ys)

let fast_resnet vs =
  let pre = conv_bn vs ~c_in:3 ~c_out:64 in
  let layer1 = layer vs ~c_in:64 ~c_out:128 in
  let inter = conv_bn vs ~c_in:128 ~c_out:256 in
  let layer2 = layer vs ~c_in:256 ~c_out:512 in
  let linear = Layer.linear vs ~use_bias:false ~input_dim:512 10 in
  Layer.of_fn_ (fun xs ~is_training ->
      let batch_size = Tensor.shape xs |> List.hd_exn in
      Layer.apply_ pre xs ~is_training
      |> Layer.apply_ layer1 ~is_training
      |> Layer.apply_ inter ~is_training
      |> Tensor.max_pool2d ~ksize:(2, 2)
      |> Layer.apply_ layer2 ~is_training
      |> Tensor.max_pool2d ~ksize:(4, 4)
      |> Tensor.reshape ~shape:[ batch_size; -1 ]
      |> Layer.apply linear
      |> fun logits -> Tensor.(logits * f 0.125))

let model vs =
  { Model.batch_size
  ; epochs
  ; lr_schedule
  ; model = fast_resnet vs
  ; model_name = "fast-resnet"
  }
