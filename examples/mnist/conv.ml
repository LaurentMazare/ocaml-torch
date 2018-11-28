open Base
open Torch

(* This should reach ~99% accuracy. *)
let batch_size = 256
let epochs = 5000
let learning_rate = 1e-3

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Some Torch_core.Device.Cuda
    end else None
  in
  let mnist = Mnist_helper.read_files () in
  let vs = Var_store.create ~name:"cnn" ?device () in
  let conv2d1 = Layer.conv2d_ vs ~ksize:5 ~stride:1 ~input_dim:1 32 in
  let conv2d2 = Layer.conv2d_ vs ~ksize:5 ~stride:1 ~input_dim:32 64 in
  let linear1 = Layer.linear vs ~activation:Relu ~input_dim:1024 1024 in
  let linear2 = Layer.linear vs ~input_dim:1024 Mnist_helper.label_count in
  let adam = Optimizer.adam vs ~learning_rate in
  let model xs ~is_training =
    Tensor.reshape xs ~shape:[ -1; 1; 28; 28 ]
    |> Layer.apply conv2d1
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Layer.apply conv2d2
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Tensor.reshape ~shape:[ -1; 1024 ]
    |> Layer.apply linear1
    |> Tensor.dropout ~p:0.5 ~is_training
    |> Layer.apply linear2
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Dataset_helper.train_batch mnist ?device ~batch_size ~batch_idx
    in
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (train_model batch_images) ~targets:batch_labels
    in

    Optimizer.backward_step adam ~loss;

    if batch_idx % 50 = 0 then begin
      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy mnist `test ?device ~batch_size ~predict:test_model
      in
      Stdio.printf "%d %f %.2f%%\n%!" batch_idx (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.full_major ();
  done
