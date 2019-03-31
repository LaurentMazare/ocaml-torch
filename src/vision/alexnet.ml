(* AlexNet model.
   https://arxiv.org/abs/1404.5997
*)
open Base
open Torch

let sub = Var_store.sub
let conv2d = Layer.conv2d_

let features vs =
  let conv1 = conv2d (sub vs "0") ~ksize:11 ~padding:2 ~stride:4 ~input_dim:3 64 in
  let conv2 = conv2d (sub vs "3") ~ksize:5 ~padding:1 ~stride:2 ~input_dim:64 192 in
  let conv3 = conv2d (sub vs "6") ~ksize:3 ~padding:1 ~stride:1 ~input_dim:192 384 in
  let conv4 = conv2d (sub vs "8") ~ksize:3 ~padding:1 ~stride:1 ~input_dim:384 256 in
  let conv5 = conv2d (sub vs "10") ~ksize:3 ~padding:1 ~stride:1 ~input_dim:256 256 in
  Layer.of_fn (fun xs ->
      Layer.apply conv1 xs
      |> Tensor.relu
      |> Tensor.max_pool2d ~ksize:(3, 3) ~stride:(2, 2)
      |> Layer.apply conv2
      |> Tensor.relu
      |> Tensor.max_pool2d ~ksize:(3, 3) ~stride:(2, 2)
      |> Layer.apply conv3
      |> Tensor.relu
      |> Layer.apply conv4
      |> Tensor.relu
      |> Layer.apply conv5
      |> Tensor.relu
      |> Tensor.max_pool2d ~ksize:(3, 3) ~stride:(2, 2) )

let classifier ?num_classes vs =
  let linear1 = Layer.linear (sub vs "1") ~input_dim:(256 * 6 * 6) 4096 in
  let linear2 = Layer.linear (sub vs "4") ~input_dim:4096 4096 in
  let linear_or_id =
    match num_classes with
    | Some num_classes -> Layer.linear (sub vs "6") ~input_dim:4096 num_classes
    | None -> Layer.id
  in
  Layer.of_fn_ (fun xs ~is_training ->
      Tensor.dropout xs ~p:0.5 ~is_training
      |> Layer.apply linear1
      |> Tensor.relu
      |> Tensor.dropout ~p:0.5 ~is_training
      |> Layer.apply linear2
      |> Tensor.relu
      |> Layer.apply linear_or_id )

let alexnet ?num_classes vs =
  let features = features (sub vs "features") in
  let classifier = classifier ?num_classes (sub vs "classifier") in
  Layer.of_fn_ (fun xs ~is_training ->
      let batch_size = Tensor.shape xs |> List.hd_exn in
      Layer.apply features xs
      |> Tensor.adaptive_avg_pool2d ~output_size:[6; 6]
      |> Tensor.view ~size:[batch_size; -1]
      |> Layer.apply_ classifier ~is_training )
