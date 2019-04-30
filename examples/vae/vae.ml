(* Variational Auto-Encoder on MNIST.
   The implementation is based on:
     https://github.com/pytorch/examples/blob/master/vae/main.py

   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz
*)

open Base
open Torch

module VAE = struct
  type t =
    { fc1 : Layer.t
    ; fc21 : Layer.t
    ; fc22 : Layer.t
    ; fc3 : Layer.t
    ; fc4 : Layer.t
    }

  let create vs =
    { fc1 = Layer.linear vs ~input_dim:784 400
    ; fc21 = Layer.linear vs ~input_dim:400 20
    ; fc22 = Layer.linear vs ~input_dim:400 20
    ; fc3 = Layer.linear vs ~input_dim:20 400
    ; fc4 = Layer.linear vs ~input_dim:400 784
    }

  let encode t xs =
    let h1 = Layer.apply t.fc1 xs |> Tensor.relu in
    Layer.apply t.fc21 h1, Layer.apply t.fc22 h1

  let decode t zs =
    Layer.apply t.fc3 zs |> Tensor.relu |> Layer.apply t.fc4 |> Tensor.sigmoid

  let forward t xs =
    let mu, logvar = encode t (Tensor.view xs ~size:[ -1; 784 ]) in
    let std_ = Tensor.(exp (logvar * f 0.5)) in
    let eps = Tensor.randn_like std_ in
    decode t Tensor.(mu + (eps * std_)), mu, logvar
end

let loss ~recon_x ~x ~mu ~logvar =
  let bce =
    Tensor.bce_loss recon_x ~targets:(Tensor.view x ~size:[ -1; 784 ]) ~reduction:Sum
  in
  let kld = Tensor.(f (-0.5) * (f 1.0 + logvar - (mu * mu) - exp logvar) |> sum) in
  Tensor.( + ) bce kld

let () =
  let device = Device.cuda_if_available () in
  let mnist = Mnist_helper.read_files () in
  let vs = Var_store.create ~name:"vae" ~device () in
  let vae = VAE.create vs in
  let opt = Optimizer.adam vs ~learning_rate:1e-3 in
  for batch_idx = 1 to 1000_000 do
    let bimages, _ = Dataset_helper.train_batch mnist ~batch_size:128 ~batch_idx in
    let bimages = Tensor.to_device bimages ~device in
    let recon_x, mu, logvar = VAE.forward vae bimages in
    let loss = loss ~recon_x ~x:bimages ~mu ~logvar in
    Optimizer.backward_step ~loss opt;
    if batch_idx % 100 = 0
    then
      Stdio.printf
        "batch %4d    loss: %12.6f\n%!"
        batch_idx
        (Tensor.float_value loss /. 128.);
    Caml.Gc.full_major ()
  done
