open Base
open Torch_tensor

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let ws =
    Tensor.zeros [Mnist_helper.image_dim; Mnist_helper.label_count]
    |> Tensor.set_requires_grad ~b:true
  in
  let bs =
    Tensor.zeros [Mnist_helper.label_count]
    |> Tensor.set_requires_grad ~b:true
  in
  let model xs = Tensor.matmul xs ws |> Tensor.add bs |> Tensor.softmax in
  for index = 1 to 1000 do
    let predicted_train_labels = model train_images in

    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.mul (Tensor.neg train_labels) (Tensor.log predicted_train_labels)
      |> Tensor.mean
    in
    Tensor.backward loss;

    (* Apply gradient descent. *)
    let wws = Tensor.set_requires_grad ~b:false ws in
    let bbs = Tensor.set_requires_grad ~b:false bs in
    Tensor.sub_assign wws (Tensor.grad ws);
    Tensor.sub_assign bbs (Tensor.grad bs);

    (* Compute validation errors. *)
    let predicted_test_labels = model test_images in
    let validation_accuracy =
      Tensor.eq
        (Tensor.argmax predicted_test_labels)
        (Tensor.argmax test_labels)
      |> Tensor.sum
      |> Tensor.float_value
      |> fun v -> v /. (Tensor.shape test_labels |> List.hd_exn |> Float.of_int)
    in
    Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. validation_accuracy);
  done
