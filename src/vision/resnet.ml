open Base
open Torch

let conv2d ?(padding = 1) ?(ksize = 3) = Layer.conv2d_ ~ksize ~padding ~use_bias:false
let sub = Var_store.sub

let downsample vs ~stride ~input_dim output_dim =
  if stride <> 1 || input_dim <> output_dim
  then (
    let conv = conv2d (sub vs "0") ~stride ~ksize:1 ~padding:0 ~input_dim output_dim in
    let bn = Layer.batch_norm2d (sub vs "1") output_dim in
    Layer.of_fn_ (fun xs ~is_training ->
        Layer.apply conv xs |> Layer.apply_ bn ~is_training))
  else Layer.id_

let basic_block vs ~stride ~input_dim output_dim =
  let conv1 = conv2d (sub vs "conv1") ~stride ~input_dim output_dim in
  let bn1 = Layer.batch_norm2d (sub vs "bn1") output_dim in
  let conv2 = conv2d (sub vs "conv2") ~stride:1 ~input_dim:output_dim output_dim in
  let bn2 = Layer.batch_norm2d (sub vs "bn2") output_dim in
  let downsample = downsample (sub vs "downsample") ~stride ~input_dim output_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv1 xs
      |> Layer.apply_ bn1 ~is_training
      |> Tensor.relu
      |> Layer.apply conv2
      |> Layer.apply_ bn2 ~is_training
      |> fun ys ->
      Tensor.( + ) ys (Layer.apply_ downsample xs ~is_training) |> Tensor.relu)

let bottleneck_block vs ~expansion ~stride ~input_dim output_dim =
  let expanded_dim = expansion * output_dim in
  let conv1 =
    conv2d (sub vs "conv1") ~stride:1 ~padding:0 ~ksize:1 ~input_dim output_dim
  in
  let bn1 = Layer.batch_norm2d (sub vs "bn1") output_dim in
  let conv2 = conv2d (sub vs "conv2") ~stride ~input_dim:output_dim output_dim in
  let bn2 = Layer.batch_norm2d (sub vs "bn2") output_dim in
  let conv3 =
    conv2d
      (sub vs "conv3")
      ~stride:1
      ~padding:0
      ~ksize:1
      ~input_dim:output_dim
      expanded_dim
  in
  let bn3 = Layer.batch_norm2d (sub vs "bn3") expanded_dim in
  let downsample = downsample (sub vs "downsample") ~stride ~input_dim expanded_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv1 xs
      |> Layer.apply_ bn1 ~is_training
      |> Tensor.relu
      |> Layer.apply conv2
      |> Layer.apply_ bn2 ~is_training
      |> Tensor.relu
      |> Layer.apply conv3
      |> Layer.apply_ bn3 ~is_training
      |> fun ys ->
      Tensor.( + ) ys (Layer.apply_ downsample xs ~is_training) |> Tensor.relu)

let resnet ?num_classes vs ~block ~layers:(c1, c2, c3, c4) =
  let block, e =
    match block with
    | `basic -> basic_block, 1
    | `bottleneck -> bottleneck_block ~expansion:4, 4
  in
  let make_layer vs ~stride ~cnt ~input_dim output_dim =
    List.init cnt ~f:(fun block_index ->
        let vs = sub vs (Int.to_string block_index) in
        if block_index = 0
        then block vs ~stride ~input_dim output_dim
        else block vs ~stride:1 ~input_dim:(output_dim * e) output_dim)
    |> Layer.fold_
  in
  let conv1 = conv2d (sub vs "conv1") ~stride:2 ~padding:3 ~ksize:7 ~input_dim:3 64 in
  let bn1 = Layer.batch_norm2d (sub vs "bn1") 64 in
  let layer1 = make_layer (sub vs "layer1") ~stride:1 ~cnt:c1 ~input_dim:64 64 in
  let layer2 = make_layer (sub vs "layer2") ~stride:2 ~cnt:c2 ~input_dim:(64 * e) 128 in
  let layer3 = make_layer (sub vs "layer3") ~stride:2 ~cnt:c3 ~input_dim:(128 * e) 256 in
  let layer4 = make_layer (sub vs "layer4") ~stride:2 ~cnt:c4 ~input_dim:(256 * e) 512 in
  let fc =
    match num_classes with
    | Some num_classes -> Layer.linear (sub vs "fc") ~input_dim:(512 * e) num_classes
    | None -> Layer.id
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let batch_size = Tensor.shape xs |> List.hd_exn in
      Layer.apply conv1 xs
      |> Layer.apply_ bn1 ~is_training
      |> Tensor.relu
      |> Tensor.max_pool2d ~stride:(2, 2) ~padding:(1, 1) ~ksize:(3, 3)
      |> Layer.apply_ layer1 ~is_training
      |> Layer.apply_ layer2 ~is_training
      |> Layer.apply_ layer3 ~is_training
      |> Layer.apply_ layer4 ~is_training
      |> Tensor.adaptive_avg_pool2d ~output_size:[ 1; 1 ]
      |> Tensor.view ~size:[ batch_size; -1 ]
      |> Layer.apply fc)

let resnet18 ?num_classes vs = resnet ?num_classes vs ~block:`basic ~layers:(2, 2, 2, 2)
let resnet34 ?num_classes vs = resnet ?num_classes vs ~block:`basic ~layers:(3, 4, 6, 3)

let resnet50 ?num_classes vs =
  resnet ?num_classes vs ~block:`bottleneck ~layers:(3, 4, 6, 3)

let resnet101 ?num_classes vs =
  resnet ?num_classes vs ~block:`bottleneck ~layers:(3, 4, 23, 3)

let resnet152 ?num_classes vs =
  resnet ?num_classes vs ~block:`bottleneck ~layers:(3, 8, 36, 3)
