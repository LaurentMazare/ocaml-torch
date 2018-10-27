(* DenseNet model for the CIFAR-10 dataset.
   https://arxiv.org/pdf/1608.06993.pdf

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This reaches xx% accuracy.
*)
open Base
open Torch

let batch_size = 32
let epochs = 300
let dropout_p = 0.2

let learning_rate ~epoch_idx =
  if epoch_idx < 150
  then 0.1
  else if epoch_idx < 225
  then 0.01
  else 0.001

let conv2d = Layer.conv2d_ ~use_bias:false

let bn_relu_conv vs ~padding ~ksize ~input_dim output_dim =
  let bn = Layer.batch_norm2d vs input_dim |> Staged.unstage in
  let conv2d = conv2d vs ~padding ~stride:1 ~ksize ~input_dim output_dim in
  fun xs ~is_training ->
    bn xs ~is_training
    |> Tensor.relu
    |> Layer.apply conv2d
    |> Tensor.dropout ~p:dropout_p ~is_training

let bottleneck vs ~growth_rate ~input_dim =
  let layer1 = bn_relu_conv vs ~padding:0 ~ksize:1 ~input_dim (4 * growth_rate) in
  let layer2 = bn_relu_conv vs ~padding:0 ~ksize:1 ~input_dim:(4 * growth_rate) growth_rate in
  fun xs ~is_training ->
    layer1 xs ~is_training
    |> layer2 ~is_training
    |> fun ys -> Tensor.cat [ xs; ys ] ~dim:1

let block_stack vs ~n ~growth_rate ~input_dim =
  let blocks =
    List.init n ~f:(fun block_index ->
      bottleneck vs ~growth_rate ~input_dim:(input_dim + growth_rate * block_index))
  in
  let output_dim = input_dim + n * growth_rate in
  output_dim,
  fun xs ~is_training ->
    List.fold blocks ~init:xs ~f:(fun acc block -> block acc ~is_training)

let densenet vs ~n1 ~n2 ~n3 ~n4 ~growth_rate =
  let interblock ~input_dim =
    let output_dim = input_dim / 2 in
    output_dim, bn_relu_conv vs ~padding:0 ~ksize:1 ~input_dim output_dim
  in
  let conv2d = conv2d vs ~padding:1 ~stride:1 ~ksize:3 ~input_dim:3 (2 * growth_rate) in
  let dim, stack1 = block_stack vs ~n:n1 ~growth_rate ~input_dim:(2 * growth_rate) in
  let dim, bn_relu_conv1 = interblock ~input_dim:dim in
  let dim, stack2 = block_stack vs ~n:n2 ~growth_rate ~input_dim:dim in
  let dim, bn_relu_conv2 = interblock ~input_dim:dim in
  let dim, stack3 = block_stack vs ~n:n3 ~growth_rate ~input_dim:dim in
  let dim, bn_relu_conv3 = interblock ~input_dim:dim in
  let dim, stack4 = block_stack vs ~n:n4 ~growth_rate ~input_dim:dim in
  let bn = Layer.batch_norm2d vs dim |> Staged.unstage in
  let linear = Layer.linear vs ~input_dim:dim Cifar_helper.label_count in
  fun xs ~is_training ->
    let batch_size = Tensor.shape xs |> List.hd_exn in
    Tensor.((xs - f 0.5) * f 4.)
    |> Tensor.reshape ~shape:Cifar_helper. [ -1; image_c; image_w; image_h ]
    |> Layer.apply conv2d
    |> stack1 ~is_training
    |> bn_relu_conv1 ~is_training
    |> Tensor.avg_pool2d ~ksize:(2, 2)
    |> stack2 ~is_training
    |> bn_relu_conv2 ~is_training
    |> Tensor.avg_pool2d ~ksize:(2, 2)
    |> stack3 ~is_training
    |> bn_relu_conv3 ~is_training
    |> Tensor.avg_pool2d ~ksize:(2, 2)
    |> stack4 ~is_training
    |> bn ~is_training
    |> Tensor.relu
    |> Tensor.avg_pool2d ~ksize:(4, 4)
    |> Tensor.reshape ~shape:[ batch_size; -1 ]
    |> Layer.apply linear

let densenet121 = densenet ~n1:6 ~n2:12 ~n3:24 ~n4:16 ~growth_rate:12

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Torch_core.Device.Cuda
    end else Cpu
  in
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Var_store.create ~name:"densenet" ~device () in
  let model = densenet121 vs in
  let sgd =
    Optimizer.sgd vs
      ~learning_rate:(learning_rate ~epoch_idx:0)
      ~momentum:0.9
      ~weight_decay:5e-4
      ~nesterov:true
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  Checkpointing.loop ~start_index:1 ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"densenet.ot"
    ~checkpoint_every:(`iters 10)
    (fun ~index:epoch_idx ->
      Optimizer.set_learning_rate sgd ~learning_rate:(learning_rate ~epoch_idx);
      let start_time = Unix.gettimeofday () in
      let sum_loss = ref 0. in
      Dataset_helper.iter cifar ~augmentation:(`flip_and_crop_with_pad 4) ~device ~batch_size
        ~f:(fun batch_idx ~batch_images ~batch_labels ->
          Optimizer.zero_grad sgd;
          let predicted = train_model batch_images in
          (* Compute the cross-entropy loss. *)
          let loss = Tensor.cross_entropy_for_logits predicted ~targets:batch_labels in
          sum_loss := !sum_loss +. Tensor.float_value loss;
          Stdio.printf "%d/%d %f\r%!"
            batch_idx
            (Dataset_helper.batches_per_epoch cifar ~batch_size)
            (!sum_loss /. Float.of_int (1 + batch_idx));
          Tensor.backward loss;
          Optimizer.step sgd);

      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy cifar `test ~device ~batch_size ~predict:test_model
      in
      Stdio.printf "%d %.0fs %f %.2f%%\n%!"
        epoch_idx
        (Unix.gettimeofday () -. start_time)
        (!sum_loss /. Float.of_int (Dataset_helper.batches_per_epoch cifar ~batch_size))
        (100. *. test_accuracy);
    )
