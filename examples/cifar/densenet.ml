(* DenseNet model for the CIFAR-10 dataset.
   https://arxiv.org/pdf/1608.06993.pdf

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This reaches 94.2% accuracy.
*)
open Base
open Torch

let sub = Var_store.sub
let batch_size = 64
let epochs = 150

let lr_schedule ~batch_idx:_ ~batches_per_epoch:_ ~epoch_idx =
  if epoch_idx < 50 then 0.1 else if epoch_idx < 150 then 0.01 else 0.001

let conv2d ?(stride = 1) ?(padding = 0) = Layer.conv2d_ ~stride ~padding ~use_bias:false

let dense_layer vs ~bn_size ~growth_rate ~input_dim =
  let inter_dim = bn_size * growth_rate in
  let bn1 = Layer.batch_norm2d (sub vs "norm1") input_dim in
  let conv1 = conv2d (sub vs "conv1") ~ksize:1 ~input_dim inter_dim in
  let bn2 = Layer.batch_norm2d (sub vs "norm2") inter_dim in
  let conv2 =
    conv2d (sub vs "conv2") ~ksize:3 ~padding:1 ~input_dim:inter_dim growth_rate
  in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply_ bn1 xs ~is_training
      |> Tensor.relu
      |> Layer.apply conv1
      |> Layer.apply_ bn2 ~is_training
      |> Tensor.relu
      |> Layer.apply conv2
      |> fun ys -> Tensor.cat [xs; ys] ~dim:1 )

let dense_block vs ~bn_size ~growth_rate ~num_layers ~input_dim =
  List.init num_layers ~f:(fun i ->
      let vs = sub vs (Printf.sprintf "denselayer%d" (1 + i)) in
      dense_layer vs ~bn_size ~growth_rate ~input_dim:(input_dim + (i * growth_rate)) )
  |> Layer.fold_

let transition vs ~input_dim output_dim =
  let bn = Layer.batch_norm2d (sub vs "norm") input_dim in
  let conv = conv2d (sub vs "conv") ~ksize:1 ~input_dim output_dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply_ bn xs ~is_training
      |> Tensor.relu
      |> Layer.apply conv
      |> Tensor.avg_pool2d ~stride:(2, 2) ~ksize:(2, 2) )

let densenet vs ~growth_rate ~block_config ~init_dim ~bn_size ~num_classes =
  let f_vs = sub vs "features" in
  let conv0 = conv2d (sub f_vs "conv0") ~ksize:3 ~padding:2 ~input_dim:3 init_dim in
  let bn0 = Layer.batch_norm2d (sub f_vs "norm0") init_dim in
  let num_features, layers =
    let last_index = List.length block_config - 1 in
    List.foldi
      block_config
      ~init:(init_dim, Layer.id_)
      ~f:(fun i (num_features, acc) num_layers ->
        let block =
          dense_block
            (sub f_vs (Printf.sprintf "denseblock%d" (1 + i)))
            ~bn_size
            ~growth_rate
            ~num_layers
            ~input_dim:num_features
        in
        let num_features = num_features + (num_layers * growth_rate) in
        if i <> last_index
        then
          let trans =
            transition
              (sub f_vs (Printf.sprintf "transition%d" (1 + i)))
              ~input_dim:num_features
              (num_features / 2)
          in
          num_features / 2, Layer.fold_ [acc; block; trans]
        else num_features, Layer.fold_ [acc; block] )
  in
  let bn5 = Layer.batch_norm2d (sub f_vs "norm5") num_features in
  let linear = Layer.linear (sub vs "classifier") ~input_dim:num_features num_classes in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.apply conv0 xs
      |> Layer.apply_ bn0 ~is_training
      |> Tensor.relu
      |> Layer.apply_ layers ~is_training
      |> Layer.apply_ bn5 ~is_training
      |> fun features ->
      Tensor.relu features
      |> Tensor.avg_pool2d ~stride:(4, 4) ~ksize:(4, 4)
      |> Tensor.view ~size:[Tensor.shape features |> List.hd_exn; -1]
      |> Layer.apply linear )

let densenet121 vs =
  densenet
    vs
    ~growth_rate:32
    ~init_dim:64
    ~block_config:[6; 12; 24; 16]
    ~bn_size:4
    ~num_classes:10

let model vs =
  { Model.model_name = "densenet121"
  ; model = densenet121 vs
  ; epochs
  ; lr_schedule
  ; batch_size }
