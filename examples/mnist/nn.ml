open Base
open Torch

(* This should reach ~97% accuracy. *)
let hidden_nodes = 128
let epochs = 1000
let learning_rate = 1e-3

let () =
  let mnist = Mnist_helper.read_files ~with_caching:true () in
  let { Mnist_helper.train_images; train_labels; _ } = mnist in
  let vs = Layer.Var_store.create ~name:"nn" () in
  let linear1 =
    Layer.linear vs hidden_nodes ~activation:Relu ~input_dim:Mnist_helper.image_dim
  in
  let linear2 =
    Layer.linear vs Mnist_helper.label_count ~activation:Softmax ~input_dim:hidden_nodes
  in
  let adam = Optimizer.adam (Layer.Var_store.vars vs) ~learning_rate in
  let model xs = Layer.apply linear1 xs |> Layer.apply linear2 in
  for index = 1 to epochs do
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (model train_images +f 1e-6))) in

    Optimizer.backward_step adam ~loss;

    if index % 50 = 0 then begin
      (* Compute the validation error. *)
      let test_accuracy =
        Mnist_helper.batch_accuracy mnist `test ~batch_size:1000 ~predict:model
      in
      Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    end;
    Caml.Gc.compact ();
  done
