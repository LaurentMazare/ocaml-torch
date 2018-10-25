(* ResNet model for the CIFAR-10 dataset.
   This uses the pre-activation variant from:
     Identity Mappings in Deep Residual Networks, Kaiming He et al. 2016.
     https://arxiv.org/pdf/1603.05027.pdf

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This reaches ~88% accuracy.
*)
open Base
open Torch

let batch_size = 128
let epochs = 150
let dropout_p = 0.3
let learning_rate ~epoch_idx =
  if epoch_idx < 50
  then 0.1
  else if epoch_idx < 100
  then 0.01
  else 0.001

let conv2d ?(padding=1) ?(ksize=3) = Layer.conv2d_ ~ksize ~padding ~use_bias:false

let basic_block vs ~stride ~input_dim output_dim =
  let conv2d1 = conv2d vs ~stride ~input_dim output_dim in
  let conv2d2 = conv2d vs ~stride:1 ~input_dim:output_dim output_dim in
  let bn1 = Layer.batch_norm2d vs output_dim |> Staged.unstage in
  let bn2 = Layer.batch_norm2d vs output_dim |> Staged.unstage in
  let shortcut =
    if stride = 1
    then fun xs ~is_training:_ -> xs
    else
      let conv = conv2d vs ~padding:0 ~ksize:1 ~stride ~input_dim output_dim in
      let bn = Layer.batch_norm2d vs output_dim |> Staged.unstage in
      fun xs ~is_training ->
        Layer.apply conv xs
        |> bn ~is_training
  in
  fun xs ~is_training ->
    Layer.apply conv2d1 xs
    |> Tensor.dropout ~p:dropout_p ~is_training
    |> bn1 ~is_training
    |> Tensor.relu
    |> Layer.apply conv2d2
    |> Tensor.dropout ~p:dropout_p ~is_training
    |> bn2 ~is_training
    |> fun ys -> Tensor.(+) ys (shortcut xs ~is_training)

let block_stack vs ~stride ~depth ~input_dim output_dim =
  let basic_blocks =
    List.init depth ~f:(fun i ->
      basic_block vs output_dim
        ~stride:(if i = 0 then stride else 1)
        ~input_dim:(if i = 0 then input_dim else output_dim))
  in
  fun (xs : Tensor.t) ~is_training ->
    List.fold basic_blocks ~init:xs
      ~f:(fun acc basic_block -> basic_block acc ~is_training)

let resnet vs =
  let conv2d = conv2d vs ~stride:1 ~input_dim:3 32 in
  let bn = Layer.batch_norm2d vs 32 |> Staged.unstage in
  let stack1 = block_stack vs ~stride:1 ~depth:2 ~input_dim:32 32 in
  let stack2 = block_stack vs ~stride:2 ~depth:2 ~input_dim:32 64 in
  let stack3 = block_stack vs ~stride:2 ~depth:2 ~input_dim:64 128 in
  let stack4 = block_stack vs ~stride:2 ~depth:2 ~input_dim:128 128 in
  let linear = Layer.linear vs ~input_dim:128 Cifar_helper.label_count in
  fun xs ~is_training ->
    let batch_size = Tensor.shape xs |> List.hd_exn in
    Tensor.((xs - f 0.5) * f 4.)
    |> Tensor.reshape ~shape:Cifar_helper. [ -1; image_c; image_w; image_h ]
    |> Layer.apply conv2d
    |> bn ~is_training
    |> Tensor.relu
    |> stack1 ~is_training
    |> stack2 ~is_training
    |> stack3 ~is_training
    |> stack4 ~is_training
    |> Tensor.avg_pool2d ~ksize:(4, 4)
    |> Tensor.reshape ~shape:[ batch_size; -1 ]
    |> Layer.apply linear

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Some Torch_core.Device.Cuda
    end else None
  in
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Var_store.create ~name:"resnet" ?device () in
  let model = resnet vs in
  let sgd =
    Optimizer.sgd vs
      ~learning_rate:(learning_rate ~epoch_idx:0)
      ~momentum:0.9
      ~weight_decay:5e-4
      ~nesterov:true
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  let batches_per_epoch = (Tensor.shape cifar.train_images |> List.hd_exn) / batch_size in
  Checkpointing.loop ~start_index:1 ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"resnet.ot"
    ~checkpoint_every:(`iters 10)
    (fun ~index:epoch_idx ->
      Optimizer.set_learning_rate sgd ~learning_rate:(learning_rate ~epoch_idx);
      let start_time = Unix.gettimeofday () in
      let sum_loss = ref 0. in
      for batch_idx = 0 to batches_per_epoch -1 do
        let batch_images, batch_labels =
          Dataset_helper.train_batch cifar ?device ~batch_size ~batch_idx
            ~augmentation:(`flip_and_crop_with_pad 4)
        in
        Optimizer.zero_grad sgd;
        let predicted = train_model batch_images in
        (* Compute the cross-entropy loss. *)
        let loss = Tensor.cross_entropy_for_logits predicted ~targets:batch_labels in
        sum_loss := !sum_loss +. Tensor.float_value loss;
        Stdio.printf "%d/%d %f\r%!" batch_idx batches_per_epoch (!sum_loss /. Float.of_int (1 + batch_idx));
        Tensor.backward loss;
        Optimizer.step sgd;
        Caml.Gc.full_major ();
      done;

      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy cifar `test ?device ~batch_size ~predict:test_model
      in
      Stdio.printf "%d %.0fs %f %.2f%%\n%!"
        epoch_idx
        (Unix.gettimeofday () -. start_time)
        (!sum_loss /. Float.of_int batches_per_epoch)
        (100. *. test_accuracy);
    )