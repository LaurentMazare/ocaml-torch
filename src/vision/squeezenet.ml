open Base
open Torch

let fire vs in_planes squeeze_planes exp1_planes exp3_planes =
  let squeeze =
    Layer.conv2d_
      (Var_store.sub vs "squeeze")
      ~ksize:1
      ~stride:1
      ~input_dim:in_planes
      squeeze_planes
  in
  let exp1 =
    Layer.conv2d_
      (Var_store.sub vs "expand1x1")
      ~ksize:1
      ~stride:1
      ~input_dim:squeeze_planes
      exp1_planes
  in
  let exp3 =
    Layer.conv2d_
      (Var_store.sub vs "expand3x3")
      ~ksize:3
      ~stride:1
      ~padding:1
      ~input_dim:exp1_planes
      exp3_planes
  in
  Layer.of_fn (fun xs ->
      let xs = Layer.apply squeeze xs |> Tensor.relu_ in
      Tensor.cat
        ~dim:1
        [Layer.apply exp1 xs |> Tensor.relu_; Layer.apply exp3 xs |> Tensor.relu_] )

let squeezenet vs ~version ~num_classes =
  let features =
    let sub_vs i = Var_store.(vs / "features" / Int.to_string i) in
    match version with
    | `v1_0 ->
      Layer.fold
        [ Layer.conv2d_ (sub_vs 0) ~ksize:7 ~stride:2 ~input_dim:3 96
        ; Layer.of_fn Tensor.relu_
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 3) 96 16 64 64
        ; fire (sub_vs 4) 128 16 64 64
        ; fire (sub_vs 5) 128 32 128 128
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 7) 256 32 128 128
        ; fire (sub_vs 8) 256 48 192 192
        ; fire (sub_vs 9) 384 48 192 192
        ; fire (sub_vs 10) 384 64 256 256
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 12) 512 64 256 256 ]
    | `v1_1 ->
      Layer.fold
        [ Layer.conv2d_ (sub_vs 0) ~ksize:3 ~stride:2 ~input_dim:3 64
        ; Layer.of_fn Tensor.relu_
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 3) 64 16 64 64
        ; fire (sub_vs 4) 128 16 64 64
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 6) 128 32 128 128
        ; fire (sub_vs 7) 256 32 128 128
        ; Layer.of_fn (Tensor.max_pool2d ~ceil_mode:true ~ksize:(3, 3) ~stride:(2, 2))
        ; fire (sub_vs 9) 256 48 192 192
        ; fire (sub_vs 10) 384 48 192 192
        ; fire (sub_vs 11) 384 64 256 256
        ; fire (sub_vs 12) 512 64 256 256 ]
  in
  let final_conv =
    Layer.conv2d_
      Var_store.(vs / "classifier" / "1")
      ~ksize:1
      ~stride:1
      ~input_dim:512
      num_classes
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let batch_size = Tensor.shape xs |> List.hd_exn in
      Layer.apply features xs
      |> Tensor.dropout ~p:0.5 ~is_training
      |> Layer.apply final_conv
      |> Tensor.relu_
      |> Tensor.adaptive_avg_pool2d ~output_size:[1; 1]
      |> Tensor.view ~size:[batch_size; num_classes] )

let squeezenet1_0 vs ~num_classes = squeezenet vs ~version:`v1_0 ~num_classes
let squeezenet1_1 vs ~num_classes = squeezenet vs ~version:`v1_1 ~num_classes
