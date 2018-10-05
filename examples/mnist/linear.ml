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
    let train_ys = model train_images in

    (* Compute the cross-entropy loss. *)
    let loss = Tensor.(mean (- train_labels * log (train_ys +f 1e-6))) in

    Tensor.backward loss;

    (* Apply gradient descent. *)
    Tensor.(no_grad ws ~f:(fun ws -> sub_assign ws (grad ws *f learning_rate)));
    Tensor.(no_grad bs ~f:(fun bs -> sub_assign bs (grad bs *f learning_rate)));

    (* Compute the validation error. *)
    let test_ys = model test_images in
    let test_accuracy =
      Tensor.(sum (argmax test_ys = argmax test_labels) |> float_value)
      |> fun sum -> sum /. Float.of_int (Tensor.shape test_images |> List.hd_exn)
    in
    Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    Caml.Gc.compact ();
  done
