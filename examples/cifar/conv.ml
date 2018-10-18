(* CNN model for the CIFAR-10 dataset.

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This reaches ~60% accuracy.
*)
open Base
open Torch

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
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Layer.Var_store.create ~name:"cnn" ?device () in
  let conv2d1 = Layer.conv2d_ vs ~ksize:5 ~stride:1 ~input_dim:3 6 in
  let conv2d2 = Layer.conv2d_ vs ~ksize:5 ~stride:1 ~input_dim:6 16 in
  let linear1 = Layer.linear vs ~activation:Relu ~input_dim:(16 * 5 * 5) 120 in
  let linear2 = Layer.linear vs ~activation:Relu ~input_dim:120 84 in
  let linear3 =
    Layer.linear vs ~activation:Softmax ~input_dim:84 Cifar_helper.label_count
  in
  let adam = Optimizer.adam (Layer.Var_store.vars vs) ~learning_rate in
  let model xs ~is_training =
    Tensor.reshape xs ~dims:Cifar_helper. [ -1; image_c; image_w; image_h ]
    |> Layer.apply conv2d1
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Layer.apply conv2d2
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Tensor.reshape ~dims:[ -1; 16 * 5 * 5 ]
    |> Layer.apply linear1
    |> Tensor.dropout ~keep_probability:0.5 ~is_training
    |> Layer.apply linear2
    |> Layer.apply linear3
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Dataset_helper.train_batch cifar ?device ~batch_size ~batch_idx
    in
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- batch_labels * log (train_model batch_images +f 1e-6))) in

    Optimizer.backward_step adam ~loss;

    if batch_idx % 50 = 0 then begin
      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy cifar `test ?device ~batch_size ~predict:test_model
      in
      Stdio.printf "%d %f %.2f%%\n%!" batch_idx (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.compact ();
  done
