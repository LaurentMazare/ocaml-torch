open Base
open Torch_tensor

(* This should reach ~97% accuracy. *)
let hidden_nodes = 128
let epochs = 1000
let learning_rate = 1e-3

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ~with_caching:true ()
  in
  let linear1 = Layer.Linear.create ~input_dim:Mnist_helper.image_dim hidden_nodes in
  let linear2 = Layer.Linear.create ~input_dim:hidden_nodes Mnist_helper.label_count in
  let adam = Optimizer.adam Layer.Linear.(vars linear1 @ vars linear2) ~learning_rate in
  let model xs =
    Layer.Linear.apply linear1 xs ~activation:Relu
    |> Layer.Linear.apply linear2 ~activation:Softmax
  in
  for index = 1 to epochs do
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (model train_images +f 1e-6))) in

    Optimizer.zero_grad adam;
    Tensor.backward loss;
    Optimizer.step adam;

    if index % 50 = 0 then begin
      (* Compute the validation error. *)
      let test_accuracy =
        Tensor.(sum (argmax (model test_images) = argmax test_labels) |> float_value)
        |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
      in
      Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.compact ();
  done
