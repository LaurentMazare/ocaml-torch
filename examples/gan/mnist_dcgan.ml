(* Deep Convolutional Generative Adverserial Networks trained on the MNIST dataset.
   https://arxiv.org/abs/1511.06434
*)
open Base
open Torch

let image_w = Mnist_helper.image_w
let image_h = Mnist_helper.image_h
let image_dim = Mnist_helper.image_dim
let latent_dim = 100

let batch_size = 128
let learning_rate = 1e-4
let batches = 10**8

let create_generator vs =
  let convt1 = Layer.ConvTranspose2D.create_ vs ~ksize:7 ~stride:1 ~padding:0 ~input_dim:100 64 in
  let convt2 = Layer.ConvTranspose2D.create_ vs ~ksize:4 ~stride:2 ~padding:1 ~input_dim:64 32 in
  let convt3 = Layer.ConvTranspose2D.create_ vs ~ksize:4 ~stride:2 ~padding:1 ~input_dim:32 1 in
  fun rand_input ->
    Layer.ConvTranspose2D.apply convt1 rand_input
    |> Tensor.const_batch_norm
    |> Tensor.relu
    |> Layer.ConvTranspose2D.apply convt2
    |> Tensor.const_batch_norm
    |> Tensor.relu
    |> Layer.ConvTranspose2D.apply convt3 ~activation:Tanh

let create_discriminator vs =
  let conv1 = Layer.Conv2D.create_ vs ~ksize:4 ~stride:2 ~padding:1 ~input_dim:1 32 in
  let conv2 = Layer.Conv2D.create_ vs ~ksize:4 ~stride:2 ~padding:1 ~input_dim:32 64 in
  let conv3 = Layer.Conv2D.create_ vs ~ksize:7 ~stride:1 ~padding:0 ~input_dim:64 1 in
  fun xs ->
    Layer.Conv2D.apply conv1 xs
    |> Tensor.leaky_relu
    |> Layer.Conv2D.apply conv2
    |> Tensor.const_batch_norm
    |> Tensor.leaky_relu
    |> Layer.Conv2D.apply conv3 ~activation:Sigmoid

let bce ?(epsilon = 1e-7) ~labels model_values =
  Tensor.(- (f labels * log (model_values + f epsilon)
    + f (1. -. labels) * log (f (1. +. epsilon) - model_values)))
  |> Tensor.mean

let rand () = Tensor.(f 2. * rand [ batch_size; latent_dim; 1; 1 ] - f 1.)

let write_samples samples ~filename =
  Stdio.Out_channel.with_file filename ~f:(fun channel ->
    Stdio.Out_channel.output_string channel "data_ = [\n";
    for sample_index = 0 to 99 do
      List.init image_dim ~f:(fun pixel_index ->
        Tensor.get_float2 samples sample_index pixel_index
        |> Printf.sprintf "%.2f")
      |> String.concat ~sep:", "
      |> Printf.sprintf "  [%s],\n"
      |> Stdio.Out_channel.output_string channel
    done;
    Stdio.Out_channel.output_string channel "]\n")

let () =
  let mnist = Mnist_helper.read_files ~with_caching:true () in

  let generator_vs = Layer.Var_store.create () in
  let generator = create_generator generator_vs in
  let opt_g = Optimizer.adam (Layer.Var_store.vars generator_vs) ~learning_rate in

  let discriminator_vs = Layer.Var_store.create () in
  let discriminator = create_discriminator discriminator_vs in
  let opt_d = Optimizer.adam (Layer.Var_store.vars discriminator_vs) ~learning_rate in

  let fixed_noise = rand () in

  for batch_idx = 1 to batches do
    let batch_images, _ = Mnist_helper.train_batch mnist ~batch_size ~batch_idx in
    let batch_images = Tensor.reshape batch_images ~dims:[ batch_size; 1; image_w; image_h ] in
    let discriminator_loss =
      Tensor.(+)
        (bce ~labels:0.9 (discriminator Tensor.(f 2. * batch_images - f 1.)))
        (bce ~labels:0.0 (rand () |> generator |> discriminator))
    in
    Optimizer.backward_step ~loss:discriminator_loss opt_d;
    let generator_loss =
      bce ~labels:1. (rand () |> generator |> discriminator)
    in
    Optimizer.backward_step ~loss:generator_loss opt_g;
    if batch_idx % 100 = 0
    then
      Stdio.printf "batch %4d    d-loss: %12.6f    g-loss: %12.6f\n%!"
        batch_idx
        (Tensor.float_value discriminator_loss)
        (Tensor.float_value generator_loss);
    Caml.Gc.compact ();
    if batch_idx % 25000 = 0 || (batch_idx < 100000 && batch_idx % 5000 = 0)
    then
      write_samples (generator fixed_noise |> Tensor.reshape ~dims:[ -1; image_dim ])
        ~filename:(Printf.sprintf "out%d.txt" batch_idx)
  done
