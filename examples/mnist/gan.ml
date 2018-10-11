(* Generative Adverserial Networks trained on the MNIST dataset. *)
open Base
open Torch

let image_dim = Mnist_helper.image_dim
let latent_dim = 100

let generator_hidden_nodes = 128
let discriminator_hidden_nodes = 128

let batch_size = 128
let learning_rate = 1e-5
let batches = 10**8

let create_generator vs =
  let linear1 = Layer.Linear.create vs ~input_dim:latent_dim generator_hidden_nodes in
  let linear2 = Layer.Linear.create vs ~input_dim:generator_hidden_nodes image_dim in
  fun rand_input ->
    Layer.Linear.apply linear1 rand_input ~activation:Leaky_relu
    |> Layer.Linear.apply linear2 ~activation:Tanh

let create_discriminator vs =
  let linear1 = Layer.Linear.create vs ~input_dim:image_dim discriminator_hidden_nodes in
  let linear2 = Layer.Linear.create vs ~input_dim:discriminator_hidden_nodes 1 in
  fun xs ->
    Layer.Linear.apply linear1 xs ~activation:Leaky_relu
    |> Layer.Linear.apply linear2 ~activation:Sigmoid

let bce ?(epsilon = 1e-7) ~labels model_values =
  Tensor.(- (labels * log (model_values + f epsilon)
    + (f 1. - labels) * log (f (1. +. epsilon) - model_values)))
  |> Tensor.mean

let rand () = Tensor.rand [ latent_dim ]

let backward_step ~loss adam =
  Optimizer.zero_grad adam;
  Tensor.backward loss;
  Optimizer.step adam

let () =
  let mnist = Mnist_helper.read_files () in

  let generator_vs = Layer.Var_store.create () in
  let generator = create_generator generator_vs in
  let opt_g = Optimizer.adam (Layer.Var_store.vars generator_vs) ~learning_rate in

  let discriminator_vs = Layer.Var_store.create () in
  let discriminator = create_discriminator discriminator_vs in
  let opt_d = Optimizer.adam (Layer.Var_store.vars discriminator_vs) ~learning_rate in

  for batch_idx = 1 to batches do
    let batch_images, _ = Mnist_helper.train_batch mnist ~batch_size ~batch_idx in
    let discriminator_loss =
      Tensor.(+)
        (bce ~labels:(Tensor.f 0.9) (discriminator batch_images))
        (bce ~labels:(Tensor.f 0.0) (rand () |> generator |> discriminator))
    in
    backward_step ~loss:discriminator_loss opt_d;
    let generator_loss =
      bce ~labels:(Tensor.f 1.) (rand () |> generator |> discriminator)
    in
    backward_step ~loss:generator_loss opt_g;
    if batch_idx % 100 = 0
    then begin
      Stdio.printf "batch %4d    d-loss: %12.6f    g-loss: %12.6f\n%!"
        batch_idx
        (Tensor.float_value discriminator_loss)
        (Tensor.float_value generator_loss);
    end
  done

