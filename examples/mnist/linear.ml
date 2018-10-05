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
open Torch_tensor

let learning_rate = 8.

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let ws = Tensor.zeros Mnist_helper. [image_dim; label_count] ~requires_grad:true in
  let bs = Tensor.zeros [Mnist_helper.label_count] ~requires_grad:true in
  let model xs = Tensor.(softmax (mm xs ws + bs)) in
  for index = 1 to 100 do
    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (model train_images +f 1e-6))) in

    Tensor.backward loss;

    (* Apply gradient descent, disable gradient tracking for these. *)
    Tensor.(no_grad ws ~f:(fun ws -> ws -= grad ws *f learning_rate));
    Tensor.(no_grad bs ~f:(fun bs -> bs -= grad bs *f learning_rate));

    (* Compute the validation error. *)
    let test_accuracy =
      Tensor.(sum (argmax (model test_images) = argmax test_labels) |> float_value)
      |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
    in
    Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    Caml.Gc.compact ();
  done
