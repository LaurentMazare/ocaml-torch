(* Linear model for the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz

   This should reach ~92% accuracy on the test dataset.
*)
open Base
open Torch

let learning_rate = Tensor.f 1.

let () =
  let { Dataset_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let ws = Tensor.zeros Mnist_helper.[ image_dim; label_count ] ~requires_grad:true in
  let bs = Tensor.zeros [ Mnist_helper.label_count ] ~requires_grad:true in
  let model xs = Tensor.(mm xs ws + bs) in
  for index = 1 to 200 do
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (model train_images) ~targets:train_labels
    in
    Tensor.backward loss;
    (* Apply gradient descent, [no_grad f] runs [f] with gradient tracking disabled. *)
    Tensor.(
      no_grad (fun () ->
          ws -= (grad ws * learning_rate);
          bs -= (grad bs * learning_rate)));
    Tensor.zero_grad ws;
    Tensor.zero_grad bs;
    (* Compute the validation error. *)
    let test_accuracy =
      Tensor.(argmax (model test_images) = test_labels)
      |> Tensor.to_kind ~kind:(T Float)
      |> Tensor.sum
      |> Tensor.float_value
      |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
    in
    Stdio.printf
      "%d %f %.2f%%\n%!"
      index
      (Tensor.float_value loss)
      (100. *. test_accuracy);
    Caml.Gc.full_major ()
  done
