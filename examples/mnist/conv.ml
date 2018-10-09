open Base
open Torch_tensor

(* This should reach ~99% accuracy. *)
let batch_size = 512
let epochs = 5000
let learning_rate = 1e-3

let () =
  let mnist = Mnist_helper.read_files ~with_caching:true () in
  let vs = Layer.Var_store.create () in
  let conv2d1 = Layer.Conv2D.create vs ~ksize:(5, 5) ~stride:(1, 1) ~input_dim:1 32 in
  let conv2d2 = Layer.Conv2D.create vs ~ksize:(5, 5) ~stride:(1, 1) ~input_dim:32 64 in
  let linear1 = Layer.Linear.create vs ~input_dim:1024 1024 in
  let linear2 = Layer.Linear.create vs ~input_dim:1024 Mnist_helper.label_count in
  let adam = Optimizer.adam (Layer.Var_store.vars vs) ~learning_rate in
  let model xs =
    Tensor.reshape xs ~dims:[ -1; 1; 28; 28 ]
    |> Layer.Conv2D.apply conv2d1
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Layer.Conv2D.apply conv2d2
    |> Tensor.max_pool2d ~ksize:(2, 2)
    |> Tensor.reshape ~dims:[ -1; 1024 ]
    |> Layer.Linear.apply linear1 ~activation:Relu
    |> Layer.Linear.apply linear2 ~activation:Softmax
  in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- batch_labels * log (model batch_images +f 1e-6))) in

    Optimizer.zero_grad adam;
    Tensor.backward loss;
    Optimizer.step adam;

    if batch_idx % 50 = 0 then begin
      (* Compute the validation error. *)
      let { Mnist_helper.test_images; test_labels; _ } = mnist in
      let test_accuracy =
        Tensor.(sum (argmax (model test_images) = argmax test_labels) |> float_value)
        |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
      in
      Stdio.printf "%d %f %.2f%%\n%!" batch_idx (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.compact ();
  done

