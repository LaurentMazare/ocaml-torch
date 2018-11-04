(* Pre-activation ResNet model for the CIFAR-10 dataset.

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This model uses the pre-activation variant of ResNet18 as introduced in:
     Identity Mappings in Deep Residual Networks
     Kaiming He et al., 2016
     https://arxiv.org/abs/1603.05027

   This reaches ~94.6% accuracy.
*)
open Base
open Torch

let batch_size = 128
let epochs = 150
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
  let bn1 = Layer.batch_norm2d vs input_dim in
  let bn2 = Layer.batch_norm2d vs output_dim in
  let shortcut =
    if stride = 1 && input_dim = output_dim
    then fun ~xs ~out:_ -> xs
    else
      let conv2d = conv2d vs ~padding:0 ~ksize:1 ~stride ~input_dim output_dim in
      fun ~xs:_ ~out -> Layer.apply conv2d out
  in
  fun xs ~is_training ->
    let out = Layer.apply_ bn1 ~is_training xs |> Tensor.relu in
    let shortcut = shortcut ~xs ~out in
    Layer.apply conv2d1 out
    |> Layer.apply_ bn2 ~is_training
    |> Tensor.relu
    |> Layer.apply conv2d2
    |> fun ys -> Tensor.(+) ys shortcut

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
  let conv2d = conv2d vs ~stride:1 ~input_dim:3 64 in
  let stack1 = block_stack vs ~stride:1 ~depth:2 ~input_dim:64 64 in
  let stack2 = block_stack vs ~stride:2 ~depth:2 ~input_dim:64 128 in
  let stack3 = block_stack vs ~stride:2 ~depth:2 ~input_dim:128 256 in
  (* The output there should be of size 512 but this requires more than
     2GB of memory.
  *)
  let stack4 = block_stack vs ~stride:2 ~depth:2 ~input_dim:256 256 in
  let linear = Layer.linear vs ~input_dim:256 Cifar_helper.label_count in
  fun xs ~is_training ->
    let batch_size = Tensor.shape xs |> List.hd_exn in
    Tensor.((xs - f 0.5) * f 4.)
    |> Tensor.reshape ~shape:Cifar_helper. [ -1; image_c; image_w; image_h ]
    |> Layer.apply conv2d
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
      Torch_core.Device.Cuda
    end else Torch_core.Device.Cpu
  in
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Var_store.create ~name:"resnet" ~device () in
  let model = resnet vs in
  let sgd =
    Optimizer.sgd vs
      ~learning_rate:(learning_rate ~epoch_idx:0)
      ~momentum:0.9
      ~weight_decay:5e-4
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  Checkpointing.loop ~start_index:1 ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"resnet.ot"
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

