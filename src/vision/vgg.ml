open Base
open Torch

let relu = Layer.of_fn_ (fun xs ~is_training:_ -> Tensor.relu xs)
let relu_ = Layer.of_fn_ (fun xs ~is_training:_ -> Tensor.relu_ xs)

type t =
  | C of int
  (* conv2d *)
  | M

(* maxpool2d *)

let layers_cfg = function
  | `A -> [ C 64; M; C 128; M; C 256; C 256; M; C 512; C 512; M; C 512; C 512; M ]
  | `B ->
    [ C 64; C 64; M; C 128; C 128; M; C 256; C 256; M; C 512; C 512; M; C 512; C 512; M ]
  | `D ->
    [ C 64
    ; C 64
    ; M
    ; C 128
    ; C 128
    ; M
    ; C 256
    ; C 256
    ; C 256
    ; M
    ; C 512
    ; C 512
    ; C 512
    ; M
    ; C 512
    ; C 512
    ; C 512
    ; M
    ]
  | `E ->
    [ C 64
    ; C 64
    ; M
    ; C 128
    ; C 128
    ; M
    ; C 256
    ; C 256
    ; C 256
    ; C 256
    ; M
    ; C 512
    ; C 512
    ; C 512
    ; C 512
    ; M
    ; C 512
    ; C 512
    ; C 512
    ; C 512
    ; M
    ]

let make_layers vs cfg ~batch_norm ~in_place_relu =
  let relu = if in_place_relu then relu_ else relu in
  let sub_vs index = Var_store.sub vs (Int.to_string index) in
  let (_output_dim, _output_idx), layers =
    List.fold_map (layers_cfg cfg) ~init:(3, 0) ~f:(fun (input_dim, idx) v ->
        match v with
        | M ->
          ( (input_dim, idx + 1)
          , [ Layer.of_fn (Tensor.max_pool2d ~ksize:(2, 2)) |> Layer.with_training ] )
        | C output_dim ->
          let conv2d =
            Layer.conv2d_ (sub_vs idx) ~ksize:3 ~stride:1 ~padding:1 ~input_dim output_dim
            |> Layer.with_training
          in
          if batch_norm
          then (
            let batch_norm = Layer.batch_norm2d (sub_vs (idx + 1)) output_dim in
            (output_dim, idx + 3), [ conv2d; batch_norm; relu ])
          else (output_dim, idx + 2), [ conv2d; relu ])
  in
  List.concat layers

let vgg ~num_classes vs cfg ~batch_norm =
  let cls_vs i = Var_store.(vs / "classifier" / Int.to_string i) in
  let layers =
    make_layers (Var_store.sub vs "features") cfg ~batch_norm ~in_place_relu:true
    |> Layer.sequential_
  in
  let fc1 = Layer.linear (cls_vs 0) ~input_dim:(512 * 7 * 7) 4096 in
  let fc2 = Layer.linear (cls_vs 3) ~input_dim:4096 4096 in
  let fc3 = Layer.linear (cls_vs 6) ~input_dim:4096 num_classes in
  Layer.of_fn_ (fun xs ~is_training ->
      let batch_size = Tensor.shape xs |> List.hd_exn in
      Layer.forward_ layers xs ~is_training
      |> Tensor.view ~size:[ batch_size; -1 ]
      |> Layer.forward fc1
      |> Tensor.relu
      |> Tensor.dropout ~p:0.5 ~is_training
      |> Layer.forward fc2
      |> Tensor.relu
      |> Tensor.dropout ~p:0.5 ~is_training
      |> Layer.forward fc3)

let vgg11 vs ~num_classes = vgg ~num_classes vs `A ~batch_norm:false
let vgg11_bn vs ~num_classes = vgg ~num_classes vs `A ~batch_norm:true
let vgg13 vs ~num_classes = vgg ~num_classes vs `B ~batch_norm:false
let vgg13_bn vs ~num_classes = vgg ~num_classes vs `B ~batch_norm:true
let vgg16 vs ~num_classes = vgg ~num_classes vs `D ~batch_norm:false
let vgg16_bn vs ~num_classes = vgg ~num_classes vs `D ~batch_norm:true
let vgg19 vs ~num_classes = vgg ~num_classes vs `E ~batch_norm:false
let vgg19_bn vs ~num_classes = vgg ~num_classes vs `E ~batch_norm:true

let vgg16_layers ?(max_layer = Int.max_value) vs ~batch_norm =
  let layers =
    List.take
      (make_layers (Var_store.sub vs "features") `D ~batch_norm ~in_place_relu:false)
      max_layer
  in
  (* [Staged.stage] just indicates that the [vs] and [~indexes] parameters should
     only be applied on the first call to this function. *)
  Staged.stage (fun xs ->
      List.fold_mapi layers ~init:xs ~f:(fun i xs layer ->
          let xs = Layer.forward_ layer xs ~is_training:false in
          xs, (i, xs))
      |> fun (_, indexed_layers) -> Map.of_alist_exn (module Int) indexed_layers)
