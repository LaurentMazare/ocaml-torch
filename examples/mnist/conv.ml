open Base
open Torch

(* This should reach ~99% accuracy. *)
let batch_size = 512
let epochs = 5000
let learning_rate = 1e-3

let () =
  let mnist = Mnist_helper.read_files ~with_caching:true () in
  let vs = Layer.Var_store.create ~name:"cnn" in
  let conv2d1 = Layer.conv2d vs ~ksize:(5, 5) ~stride:(1, 1) ~input_dim:1 32 in
  let conv2d2 = Layer.conv2d vs ~ksize:(5, 5) ~stride:(1, 1) ~input_dim:32 64 in
  let linear1 = Layer.linear vs ~activation:Relu ~input_dim:1024 1024 in
  let linear2 =
    Layer.linear vs ~activation:Softmax ~input_dim:1024 Mnist_helper.label_count
  in
  let adam = Optimizer.adam (Layer.Var_store.vars vs) ~learning_rate in
  let model xs ~is_training =
    Tensor.reshape xs ~dims:[ -1; 1; 28; 28 ]
    |> Layer.apply conv2d1
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Layer.apply conv2d2
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Tensor.reshape ~dims:[ -1; 1024 ]
    |> Layer.apply linear1
    |> Tensor.dropout ~keep_probability:0.5 ~is_training
    |> Layer.apply linear2
  in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- batch_labels * log (train_model batch_images +f 1e-6))) in

    Optimizer.backward_step adam ~loss;

    if batch_idx % 50 = 0 then begin
      (* Compute the validation error. *)
      let test_accuracy =
        Mnist_helper.batch_accuracy mnist `test ~batch_size:1000 ~predict:test_model
      in
      Stdio.printf "%d %f %.2f%%\n%!" batch_idx (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.compact ();
  done

