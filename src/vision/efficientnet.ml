(* EfficientNet models, https://arxiv.org/abs/1905.11946
   The implementation is very similar to:
     https://github.com/lukemelas/EfficientNet-PyTorch
*)
open Base
open Torch

let batch_norm2d = Layer.batch_norm2d ~momentum:0.01 ~eps:1e-3

type block_args =
  { kernel_size : int
  ; num_repeat : int
  ; input_filters : int
  ; output_filters : int
  ; expand_ratio : int
  ; se_ratio : float
  ; stride : int
  }

let block_args () =
  [ 3, 1, 32, 16, 1, 0.25, 1
  ; 3, 2, 16, 24, 6, 0.25, 2
  ; 5, 2, 24, 40, 6, 0.25, 2
  ; 3, 3, 40, 80, 6, 0.25, 2
  ; 5, 3, 80, 112, 6, 0.25, 1
  ; 5, 4, 112, 192, 6, 0.25, 2
  ; 3, 1, 192, 320, 6, 0.25, 1
  ]
  |> List.map ~f:(fun (k, n, i, o, e, se, s) ->
         { kernel_size = k
         ; num_repeat = n
         ; input_filters = i
         ; output_filters = o
         ; expand_ratio = e
         ; se_ratio = se
         ; stride = s
         })

type params =
  { width : float
  ; depth : float
  ; res : int
  ; dropout : float
  }

let round_repeats params repeats =
  Float.iround_up_exn (params.depth *. Float.of_int repeats)

let round_filters params filters =
  let divisor = 8 in
  let filters = params.width *. Float.of_int filters in
  let filters_ = Float.to_int (filters +. (Float.of_int divisor /. 2.)) in
  let new_filters = Int.max divisor (filters_ / divisor * divisor) in
  if Float.(of_int new_filters < 0.9 * filters)
  then new_filters + divisor
  else new_filters

(* Conv2D with same padding *)
let conv2d vs ?(use_bias = false) ?(ksize = 1) ?(stride = 1) ?(groups = 1) ~input_dim dim =
  let conv2d = Layer.conv2d_ vs ~ksize ~stride ~groups ~use_bias ~input_dim dim in
  Layer.of_fn_ (fun xs ~is_training:_ ->
      let _, _, ih, iw = Tensor.shape4_exn xs in
      let oh = (ih + stride - 1) / stride in
      let ow = (iw + stride - 1) / stride in
      let pad_h = Int.max 0 (((oh - 1) * stride) + ksize - ih) in
      let pad_w = Int.max 0 (((ow - 1) * stride) + ksize - iw) in
      let xs =
        if pad_h > 0 || pad_w > 0
        then
          Tensor.constant_pad_nd
            xs
            ~pad:[ pad_w / 2; pad_w - (pad_w / 2); pad_h / 2; pad_h - (pad_h / 2) ]
        else xs
      in
      Layer.forward conv2d xs)

let swish xs = Tensor.(xs * sigmoid xs)
let swish_layer = Layer.of_fn_ (fun xs ~is_training:_ -> swish xs)

let block vs args =
  let vs = Var_store.sub vs in
  let inp = args.input_filters in
  let oup = args.input_filters * args.expand_ratio in
  let final_oup = args.output_filters in
  let expansion =
    if args.expand_ratio <> 1
    then
      Layer.sequential_
        [ conv2d (vs "_expand_conv") ~input_dim:inp oup
        ; batch_norm2d (vs "_bn0") oup
        ; swish_layer
        ]
    else Layer.id_
  in
  let depthwise_conv =
    conv2d
      (vs "_depthwise_conv")
      ~ksize:args.kernel_size
      ~groups:oup
      ~stride:args.stride
      ~input_dim:oup
      oup
  in
  let depthwise_bn = batch_norm2d (vs "_bn1") oup in
  let se =
    let nsc = Int.max 1 (Float.to_int (args.se_ratio *. Float.of_int inp)) in
    Layer.sequential_
      [ conv2d (vs "_se_reduce") ~use_bias:true ~input_dim:oup nsc
      ; swish_layer
      ; conv2d (vs "_se_expand") ~use_bias:true ~input_dim:nsc oup
      ]
  in
  let project_conv = conv2d (vs "_project_conv") ~input_dim:oup final_oup in
  let project_bn = batch_norm2d (vs "_bn2") final_oup in
  Layer.of_fn_ (fun xs ~is_training ->
      let ys =
        Layer.forward_ expansion xs ~is_training
        |> Layer.forward_ depthwise_conv ~is_training
        |> Layer.forward_ depthwise_bn ~is_training
        |> swish
      in
      let zs =
        Tensor.adaptive_avg_pool2d ys ~output_size:[ 1; 1 ]
        |> Layer.forward_ se ~is_training
      in
      let ys =
        Tensor.(ys * sigmoid zs)
        |> Layer.forward_ project_conv ~is_training
        |> Layer.forward_ project_bn ~is_training
      in
      if args.stride = 1 && inp = final_oup then Tensor.(xs + ys) else ys)

let efficientnet ?(num_classes = 1000) vs params =
  let args = block_args () in
  let vs = Var_store.sub vs in
  let out_c = round_filters params 32 in
  let conv_stem = conv2d (vs "_conv_stem") ~ksize:3 ~input_dim:3 ~stride:2 out_c in
  let bn0 = batch_norm2d (vs "_bn0") out_c in
  let blocks =
    let vs = vs "_blocks" in
    let blocks = ref [] in
    let idx = ref 0 in
    let add_block ba =
      let vs = Var_store.sub vs (Int.to_string !idx) in
      Int.incr idx;
      blocks := block vs ba :: !blocks
    in
    List.iter args ~f:(fun arg ->
        let arg =
          { arg with
            input_filters = round_filters params arg.input_filters
          ; output_filters = round_filters params arg.output_filters
          }
        in
        add_block arg;
        let arg = { arg with input_filters = arg.output_filters; stride = 1 } in
        for _i = 2 to round_repeats params arg.num_repeat do
          add_block arg
        done);
    List.rev !blocks |> Layer.sequential_
  in
  let out_c = round_filters params 1280 in
  let input_dim = (List.last_exn args).output_filters |> round_filters params in
  let conv_head = conv2d (vs "_conv_head") ~input_dim out_c in
  let bn1 = batch_norm2d (vs "_bn1") out_c in
  let fc = Layer.linear (vs "_fc") ~input_dim:out_c num_classes in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.forward_ conv_stem xs ~is_training
      |> Layer.forward_ bn0 ~is_training
      |> swish
      |> Layer.forward_ blocks ~is_training
      |> Layer.forward_ conv_head ~is_training
      |> Layer.forward_ bn1 ~is_training
      |> swish
      |> Tensor.adaptive_avg_pool2d ~output_size:[ 1; 1 ]
      |> Tensor.squeeze_dim ~dim:(-1)
      |> Tensor.squeeze_dim ~dim:(-1)
      |> Tensor.dropout ~p:0.2 ~is_training
      |> Layer.forward fc)

let b0 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.0; depth = 1.0; res = 224; dropout = 0.2 }

let b1 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.0; depth = 1.1; res = 240; dropout = 0.2 }

let b2 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.1; depth = 1.2; res = 260; dropout = 0.3 }

let b3 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.2; depth = 1.4; res = 300; dropout = 0.3 }

let b4 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.4; depth = 1.8; res = 380; dropout = 0.4 }

let b5 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.6; depth = 2.2; res = 456; dropout = 0.4 }

let b6 ?num_classes vs =
  efficientnet ?num_classes vs { width = 1.8; depth = 2.6; res = 528; dropout = 0.5 }

let b7 ?num_classes vs =
  efficientnet ?num_classes vs { width = 2.0; depth = 3.1; res = 600; dropout = 0.5 }
