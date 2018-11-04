open Base
open Torch

let conv2d ?(padding=1) ?(ksize=3) = Layer.conv2d_ ~ksize ~padding ~use_bias:false

let _basic_block vs ?(stride=1) ~input_dim output_dim =
  let conv1 = conv2d vs ~stride ~input_dim output_dim in
  let bn1 = Layer.batch_norm2d vs output_dim in
  let conv2 = conv2d vs ~stride:1 ~input_dim:output_dim output_dim in
  let bn2 = Layer.batch_norm2d vs output_dim in
  let downsample =
    if stride <> 1 || input_dim <> output_dim
    then
      let conv = conv2d vs ~stride ~ksize:1 ~input_dim output_dim in
      let bn = Layer.batch_norm2d vs output_dim in
      Layer.of_fn_ (fun xs ~is_training ->
        Layer.apply conv xs |> Layer.apply_ bn ~is_training)
    else Layer.id_
  in
  Layer.of_fn_ (fun xs ~is_training ->
    Layer.apply conv1 xs
    |> Layer.apply_ bn1 ~is_training
    |> Tensor.relu
    |> Layer.apply conv2
    |> Layer.apply_ bn2 ~is_training
    |> fun ys -> Tensor.(+) ys (Layer.apply_ downsample xs ~is_training))

