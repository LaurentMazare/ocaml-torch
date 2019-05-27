(* Inception V3 *)
open Base
open Torch

let sub = Var_store.sub
let max_pool2d ~k ~stride:s = Tensor.max_pool2d ~ksize:(k, k) ~stride:(s, s)

let cv_bn vs ~k ~pad ?(stride = 1) ~input_dim out_dim =
  let conv =
    Layer.conv2d_
      (sub vs "conv")
      ~ksize:k
      ~padding:pad
      ~stride
      ~use_bias:false
      ~input_dim
      out_dim
  in
  let bn = Layer.batch_norm2d (sub vs "bn") ~eps:0.001 out_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv xs |> Layer.apply_ bn ~is_training |> Tensor.relu)

let cv_bn2 vs ~k ~pad ?(stride = 1, 1) ~input_dim out_dim =
  let conv =
    Layer.conv2d
      (sub vs "conv")
      ~ksize:k
      ~padding:pad
      ~stride
      ~use_bias:false
      ~input_dim
      out_dim
  in
  let bn = Layer.batch_norm2d (sub vs "bn") ~eps:0.001 out_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv xs |> Layer.apply_ bn ~is_training |> Tensor.relu)

let inception_a vs ~input_dim ~pool =
  let b1 = cv_bn (sub vs "branch1x1") ~k:1 ~pad:0 ~input_dim 64 in
  let b2_1 = cv_bn (sub vs "branch5x5_1") ~k:1 ~pad:0 ~input_dim 48 in
  let b2_2 = cv_bn (sub vs "branch5x5_2") ~k:5 ~pad:2 ~input_dim:48 64 in
  let b3_1 = cv_bn (sub vs "branch3x3dbl_1") ~k:1 ~pad:0 ~input_dim 64 in
  let b3_2 = cv_bn (sub vs "branch3x3dbl_2") ~k:3 ~pad:1 ~input_dim:64 96 in
  let b3_3 = cv_bn (sub vs "branch3x3dbl_3") ~k:3 ~pad:1 ~input_dim:96 96 in
  let bpool = cv_bn (sub vs "branch_pool") ~k:1 ~pad:0 ~input_dim pool in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let b1 = apply_ b1 xs in
      let b2 = apply_ b2_1 xs |> apply_ b2_2 in
      let b3 = apply_ b3_1 xs |> apply_ b3_2 |> apply_ b3_3 in
      let bpool =
        Tensor.avg_pool2d xs ~ksize:(3, 3) ~stride:(1, 1) ~padding:(1, 1) |> apply_ bpool
      in
      Tensor.cat [ b1; b2; b3; bpool ] ~dim:1)

let inception_b vs ~input_dim =
  let b1 = cv_bn (sub vs "branch3x3") ~k:3 ~pad:0 ~stride:2 ~input_dim 384 in
  let b2_1 = cv_bn (sub vs "branch3x3dbl_1") ~k:1 ~pad:0 ~input_dim 64 in
  let b2_2 = cv_bn (sub vs "branch3x3dbl_2") ~k:3 ~pad:1 ~input_dim:64 96 in
  let b2_3 = cv_bn (sub vs "branch3x3dbl_3") ~k:3 ~pad:0 ~stride:2 ~input_dim:96 96 in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let b1 = apply_ b1 xs in
      let b2 = apply_ b2_1 xs |> apply_ b2_2 |> apply_ b2_3 in
      let bpool = max_pool2d xs ~k:3 ~stride:2 in
      Tensor.cat [ b1; b2; bpool ] ~dim:1)

let inception_c vs ~input_dim ~c7 =
  let b1 = cv_bn (sub vs "branch1x1") ~k:1 ~pad:0 ~input_dim 192 in
  let b2_1 = cv_bn (sub vs "branch7x7_1") ~k:1 ~pad:0 ~input_dim c7 in
  let b2_2 = cv_bn2 (sub vs "branch7x7_2") ~k:(1, 7) ~pad:(0, 3) ~input_dim:c7 c7 in
  let b2_3 = cv_bn2 (sub vs "branch7x7_3") ~k:(7, 1) ~pad:(3, 0) ~input_dim:c7 192 in
  let b3_1 = cv_bn (sub vs "branch7x7dbl_1") ~k:1 ~pad:0 ~input_dim c7 in
  let b3_2 = cv_bn2 (sub vs "branch7x7dbl_2") ~k:(7, 1) ~pad:(3, 0) ~input_dim:c7 c7 in
  let b3_3 = cv_bn2 (sub vs "branch7x7dbl_3") ~k:(1, 7) ~pad:(0, 3) ~input_dim:c7 c7 in
  let b3_4 = cv_bn2 (sub vs "branch7x7dbl_4") ~k:(7, 1) ~pad:(3, 0) ~input_dim:c7 c7 in
  let b3_5 = cv_bn2 (sub vs "branch7x7dbl_5") ~k:(1, 7) ~pad:(0, 3) ~input_dim:c7 192 in
  let bpool = cv_bn (sub vs "branch_pool") ~k:1 ~pad:0 ~input_dim 192 in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let b1 = apply_ b1 xs in
      let b2 = apply_ b2_1 xs |> apply_ b2_2 |> apply_ b2_3 in
      let b3 =
        apply_ b3_1 xs |> apply_ b3_2 |> apply_ b3_3 |> apply_ b3_4 |> apply_ b3_5
      in
      let bpool =
        Tensor.avg_pool2d xs ~ksize:(3, 3) ~stride:(1, 1) ~padding:(1, 1) |> apply_ bpool
      in
      Tensor.cat [ b1; b2; b3; bpool ] ~dim:1)

let inception_d vs ~input_dim =
  let b1_1 = cv_bn (sub vs "branch3x3_1") ~k:1 ~pad:0 ~input_dim 192 in
  let b1_2 = cv_bn (sub vs "branch3x3_2") ~k:3 ~pad:0 ~stride:2 ~input_dim:192 320 in
  let b2_1 = cv_bn (sub vs "branch7x7x3_1") ~k:1 ~pad:0 ~input_dim 192 in
  let b2_2 = cv_bn2 (sub vs "branch7x7x3_2") ~k:(1, 7) ~pad:(0, 3) ~input_dim:192 192 in
  let b2_3 = cv_bn2 (sub vs "branch7x7x3_3") ~k:(7, 1) ~pad:(3, 0) ~input_dim:192 192 in
  let b2_4 = cv_bn (sub vs "branch7x7x3_4") ~k:3 ~pad:0 ~stride:2 ~input_dim:192 192 in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let b1 = apply_ b1_1 xs |> apply_ b1_2 in
      let b2 = apply_ b2_1 xs |> apply_ b2_2 |> apply_ b2_3 |> apply_ b2_4 in
      let bpool = max_pool2d xs ~k:3 ~stride:2 in
      Tensor.cat [ b1; b2; bpool ] ~dim:1)

let inception_e vs ~input_dim =
  let b1 = cv_bn (sub vs "branch1x1") ~k:1 ~pad:0 ~input_dim 320 in
  let b2_1 = cv_bn (sub vs "branch3x3_1") ~k:1 ~pad:0 ~input_dim 384 in
  let b2_2a = cv_bn2 (sub vs "branch3x3_2a") ~k:(1, 3) ~pad:(0, 1) ~input_dim:384 384 in
  let b2_2b = cv_bn2 (sub vs "branch3x3_2b") ~k:(3, 1) ~pad:(1, 0) ~input_dim:384 384 in
  let b3_1 = cv_bn (sub vs "branch3x3dbl_1") ~k:1 ~pad:0 ~input_dim 448 in
  let b3_2 = cv_bn (sub vs "branch3x3dbl_2") ~k:3 ~pad:1 ~input_dim:448 384 in
  let b3_3a =
    cv_bn2 (sub vs "branch3x3dbl_3a") ~k:(1, 3) ~pad:(0, 1) ~input_dim:384 384
  in
  let b3_3b =
    cv_bn2 (sub vs "branch3x3dbl_3b") ~k:(3, 1) ~pad:(1, 0) ~input_dim:384 384
  in
  let bpool = cv_bn (sub vs "branch_pool") ~k:1 ~pad:0 ~input_dim 192 in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let b1 = apply_ b1 xs in
      let b2 = apply_ b2_1 xs in
      let b2 = Tensor.cat [ apply_ b2_2a b2; apply_ b2_2b b2 ] ~dim:1 in
      let b3 = apply_ b3_1 xs |> apply_ b3_2 in
      let b3 = Tensor.cat [ apply_ b3_3a b3; apply_ b3_3b b3 ] ~dim:1 in
      let bpool =
        Tensor.avg_pool2d xs ~ksize:(3, 3) ~stride:(1, 1) ~padding:(1, 1) |> apply_ bpool
      in
      Tensor.cat [ b1; b2; b3; bpool ] ~dim:1)

let v3 ?num_classes vs =
  let conv1 = cv_bn (sub vs "Conv2d_1a_3x3") ~k:3 ~pad:0 ~stride:2 ~input_dim:3 32 in
  let conv2a = cv_bn (sub vs "Conv2d_2a_3x3") ~k:3 ~pad:0 ~input_dim:32 32 in
  let conv2b = cv_bn (sub vs "Conv2d_2b_3x3") ~k:3 ~pad:1 ~input_dim:32 64 in
  let conv3b = cv_bn (sub vs "Conv2d_3b_1x1") ~k:1 ~pad:0 ~input_dim:64 80 in
  let conv4a = cv_bn (sub vs "Conv2d_4a_3x3") ~k:3 ~pad:0 ~input_dim:80 192 in
  let mixed_5b = inception_a (sub vs "Mixed_5b") ~input_dim:192 ~pool:32 in
  let mixed_5c = inception_a (sub vs "Mixed_5c") ~input_dim:256 ~pool:64 in
  let mixed_5d = inception_a (sub vs "Mixed_5d") ~input_dim:288 ~pool:64 in
  let mixed_6a = inception_b (sub vs "Mixed_6a") ~input_dim:288 in
  let mixed_6b = inception_c (sub vs "Mixed_6b") ~input_dim:768 ~c7:128 in
  let mixed_6c = inception_c (sub vs "Mixed_6c") ~input_dim:768 ~c7:160 in
  let mixed_6d = inception_c (sub vs "Mixed_6d") ~input_dim:768 ~c7:160 in
  let mixed_6e = inception_c (sub vs "Mixed_6e") ~input_dim:768 ~c7:192 in
  let mixed_7a = inception_d (sub vs "Mixed_7a") ~input_dim:768 in
  let mixed_7b = inception_e (sub vs "Mixed_7b") ~input_dim:1280 in
  let mixed_7c = inception_e (sub vs "Mixed_7c") ~input_dim:2048 in
  let final =
    match num_classes with
    | None -> Layer.id
    | Some num_classes -> Layer.linear (sub vs "fc") ~input_dim:2048 num_classes
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let apply_ = Layer.apply_ ~is_training in
      let batch_size = Tensor.shape xs |> List.hd_exn in
      apply_ conv1 xs
      |> apply_ conv2a
      |> apply_ conv2b
      |> Tensor.relu
      |> max_pool2d ~k:3 ~stride:2
      |> apply_ conv3b
      |> apply_ conv4a
      |> max_pool2d ~k:3 ~stride:2
      |> apply_ mixed_5b
      |> apply_ mixed_5c
      |> apply_ mixed_5d
      |> apply_ mixed_6a
      |> apply_ mixed_6b
      |> apply_ mixed_6c
      |> apply_ mixed_6d
      |> apply_ mixed_6e
      |> apply_ mixed_7a
      |> apply_ mixed_7b
      |> apply_ mixed_7c
      |> Tensor.adaptive_avg_pool2d ~output_size:[ 1; 1 ]
      |> Tensor.dropout ~p:0.5 ~is_training
      |> Tensor.view ~size:[ batch_size; -1 ]
      |> Layer.apply final)
