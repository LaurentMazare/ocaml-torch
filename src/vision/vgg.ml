open Base
open Torch

let relu_ = Layer.of_fn_ (fun xs ~is_training:_ -> Tensor.relu_ xs)

type t =
  | C of int (* conv2d *)
  | M        (* maxpool2d *)

let layers_cfg = function
  | `A -> [ C 64; M; C 128; M; C 256; C 256; M; C 512; C 512; M; C 512; C 512; M ]
  | `B ->
    [ C 64; C 64; M; C 128; C 128; M; C 256; C 256; M; C 512; C 512; M; C 512; C 512; M ]
  | `D ->
    [ C 64; C 64; M
    ; C 128; C 128; M
    ; C 256; C 256; C 256; M
    ; C 512; C 512; C 512; M
    ; C 512; C 512; C 512; M
    ]
  | `E ->
    [ C 64; C 64; M
    ; C 128; C 128; M
    ; C 256; C 256; C 256; C 256; M
    ; C 512; C 512; C 512; C 512; M
    ; C 512; C 512; C 512; C 512; M
    ]

let make_layers vs cfg ~n ~batch_norm =
  let n index = N.(n / Int.to_string index) in
  let _output_dim, layers =
    List.fold (layers_cfg cfg) ~init:(3, []) ~f:(fun (input_dim, acc) v ->
        let idx = List.length acc in
        let layer, output_dim =
          match v with
          | M ->
            [ Layer.of_fn (Tensor.max_pool2d ~ksize:(2, 2)) |> Layer.with_training ],
            input_dim
          | C output_dim ->
            let conv2d =
              Layer.conv2d_ vs ~n:(n idx) ~ksize:3 ~stride:1 ~padding:1 ~input_dim output_dim
              |> Layer.with_training
            in
            if batch_norm
            then
              let batch_norm = Layer.batch_norm2d vs ~n:(n (idx+1)) output_dim in
              [ conv2d; batch_norm; relu_ ], output_dim
            else [ conv2d; relu_ ], output_dim
        in
        output_dim, layer @ acc)
  in
  List.rev layers |> Layer.fold_

let vgg ~num_classes vs cfg ~batch_norm =
  let n str = N.(root / str) in
  let cls i = N.(root / "classifier" / Int.to_string i) in
  let layers = make_layers vs cfg ~n:(n "features") ~batch_norm in
  let fc1 = Layer.linear vs ~n:(cls 0) ~input_dim:(512 * 7 * 7) 4096 in
  let fc2 = Layer.linear vs ~n:(cls 3) ~input_dim:4096 4096 in
  let fc3 = Layer.linear vs ~n:(cls 6) ~input_dim:4096 num_classes in
  Layer.of_fn_ (fun xs ~is_training ->
    let batch_size = Tensor.shape xs |> List.hd_exn in
    Layer.apply_ layers xs ~is_training
    |> Tensor.view ~size:[ batch_size; -1 ]
    |> Layer.apply fc1
    |> Tensor.relu
    |> Tensor.dropout ~p:0.5 ~is_training
    |> Layer.apply fc2
    |> Tensor.relu
    |> Tensor.dropout ~p:0.5 ~is_training
    |> Layer.apply fc3)

let vgg11 vs ~num_classes = vgg ~num_classes vs `A ~batch_norm:false
let vgg11_bn vs ~num_classes = vgg ~num_classes vs `A ~batch_norm:true
let vgg13 vs ~num_classes = vgg ~num_classes vs `B ~batch_norm:false
let vgg13_bn vs ~num_classes = vgg ~num_classes vs `B ~batch_norm:true
let vgg16 vs ~num_classes = vgg ~num_classes vs `D ~batch_norm:false
let vgg16_bn vs ~num_classes = vgg ~num_classes vs `D ~batch_norm:true
let vgg19 vs ~num_classes = vgg ~num_classes vs `E ~batch_norm:false
let vgg19_bn vs ~num_classes = vgg ~num_classes vs `E ~batch_norm:true
