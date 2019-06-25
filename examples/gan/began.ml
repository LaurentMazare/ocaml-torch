(* Boundary Equilibrium GAN.
   https://arxiv.org/abs/1703.10717
*)
open Base
open Torch

let img_size = 128
let batch_size = 16
let learning_rate = 1e-4
let lr_update_step = 3000.
let batches = 10 ** 8
let gamma = 0.5
let lambda_k = 0.001
let latent_dim = 64
let num_channel = 64
let conv2d ?(ksize = 3) ?(padding = 1) vs = Layer.conv2d_ ~ksize ~stride:1 ~padding vs

let upsample xs =
  let _, _, x, y = Tensor.shape4_exn xs in
  Tensor.upsample_nearest2d xs ~output_size:[ 2 * x; 2 * y ]

let avg_pool2d = Tensor.avg_pool2d ~ksize:(2, 2) ~stride:(2, 2)

let decoder vs =
  let l0 = Layer.linear vs ~input_dim:latent_dim (8 * 8 * num_channel) in
  let l1 = conv2d vs ~input_dim:num_channel num_channel in
  let l2 = conv2d vs ~input_dim:num_channel num_channel in
  let l3 = conv2d vs ~input_dim:num_channel num_channel in
  let l4 = conv2d vs ~input_dim:num_channel num_channel in
  let l5 = conv2d vs ~input_dim:num_channel num_channel in
  let l6 = conv2d vs ~input_dim:num_channel num_channel in
  let l7 = conv2d vs ~input_dim:num_channel num_channel in
  let l8 = conv2d vs ~input_dim:num_channel num_channel in
  let l9 = conv2d vs ~input_dim:num_channel num_channel in
  let l10 = conv2d vs ~input_dim:num_channel num_channel in
  let l11 = conv2d vs ~input_dim:num_channel 3 in
  Layer.of_fn (fun xs ->
      Tensor.to_device xs ~device:(Var_store.device vs)
      |> Layer.forward l0
      |> Tensor.view ~size:[ batch_size; num_channel; 8; 8 ]
      |> Layer.forward l1
      |> Tensor.elu
      |> Layer.forward l2
      |> Tensor.elu
      |> upsample
      |> Layer.forward l3
      |> Tensor.elu
      |> Layer.forward l4
      |> Tensor.elu
      |> upsample
      |> Layer.forward l5
      |> Tensor.elu
      |> Layer.forward l6
      |> Tensor.elu
      |> upsample
      |> Layer.forward l7
      |> Tensor.elu
      |> Layer.forward l8
      |> Tensor.elu
      |> upsample
      |> Layer.forward l9
      |> Tensor.elu
      |> Layer.forward l10
      |> Tensor.elu
      |> Layer.forward l11
      |> Tensor.tanh)

let encoder vs =
  let l0 = conv2d vs ~input_dim:3 num_channel in
  let l1 = conv2d vs ~input_dim:num_channel num_channel in
  let l2 = conv2d vs ~input_dim:num_channel num_channel in
  let down1 = conv2d vs ~ksize:1 ~padding:0 ~input_dim:num_channel num_channel in
  let l3 = conv2d vs ~input_dim:num_channel num_channel in
  let l4 = conv2d vs ~input_dim:num_channel num_channel in
  let down2 = conv2d vs ~ksize:1 ~padding:0 ~input_dim:num_channel (2 * num_channel) in
  let l5 = conv2d vs ~input_dim:(2 * num_channel) (2 * num_channel) in
  let l6 = conv2d vs ~input_dim:(2 * num_channel) (2 * num_channel) in
  let down3 =
    conv2d vs ~ksize:1 ~padding:0 ~input_dim:(2 * num_channel) (3 * num_channel)
  in
  let l7 = conv2d vs ~input_dim:(3 * num_channel) (3 * num_channel) in
  let l8 = conv2d vs ~input_dim:(3 * num_channel) (3 * num_channel) in
  let down4 =
    conv2d vs ~ksize:1 ~padding:0 ~input_dim:(3 * num_channel) (4 * num_channel)
  in
  let l9 = conv2d vs ~input_dim:(4 * num_channel) (4 * num_channel) in
  let l10 = conv2d vs ~input_dim:(4 * num_channel) (4 * num_channel) in
  let l11 = Layer.linear vs ~input_dim:(8 * 8 * 4 * num_channel) latent_dim in
  Layer.of_fn (fun xs ->
      Tensor.to_device xs ~device:(Var_store.device vs)
      |> Layer.forward l0
      |> Tensor.elu
      |> Layer.forward l1
      |> Tensor.elu
      |> Layer.forward l2
      |> Tensor.elu
      |> Layer.forward down1
      |> avg_pool2d
      |> Layer.forward l3
      |> Tensor.elu
      |> Layer.forward l4
      |> Tensor.elu
      |> Layer.forward down2
      |> avg_pool2d
      |> Layer.forward l5
      |> Tensor.elu
      |> Layer.forward l6
      |> Tensor.elu
      |> Layer.forward down3
      |> avg_pool2d
      |> Layer.forward l7
      |> Tensor.elu
      |> Layer.forward l8
      |> Tensor.elu
      |> Layer.forward down4
      |> avg_pool2d
      |> Layer.forward l9
      |> Tensor.elu
      |> Layer.forward l10
      |> Tensor.elu
      |> Tensor.view ~size:[ batch_size; 8 * 8 * 4 * num_channel ]
      |> Layer.forward l11
      |> Tensor.elu)

let create_discriminator vs =
  let encoder = encoder vs in
  let decoder = decoder vs in
  Layer.of_fn (fun xs -> Layer.forward encoder xs |> Layer.forward decoder)

let z_dist () = Tensor.((rand [ batch_size; latent_dim ] * f 2.) - f 1.)

let write_samples samples ~filename =
  List.init 4 ~f:(fun i ->
      List.init 4 ~f:(fun j ->
          Tensor.narrow samples ~dim:0 ~start:((4 * i) + j) ~length:1)
      |> Tensor.cat ~dim:2)
  |> Tensor.cat ~dim:3
  |> Torch_vision.Image.write_image ~filename

let () =
  let device = Device.cuda_if_available () in
  if Array.length Sys.argv < 2
  then Printf.failwithf "Usage: %s images.ot" Sys.argv.(0) ();
  let images = Serialize.load ~filename:Sys.argv.(1) in
  let train_size = Tensor.shape images |> List.hd_exn in
  let generator_vs = Var_store.create ~name:"gen" ~device () in
  let generator = decoder generator_vs in
  let opt_g = Optimizer.adam generator_vs ~learning_rate ~beta1:0.5 ~beta2:0.999 in
  let discriminator_vs = Var_store.create ~name:"disc" ~device () in
  let discriminator = create_discriminator discriminator_vs in
  let opt_d = Optimizer.adam discriminator_vs ~learning_rate ~beta1:0.5 ~beta2:0.999 in
  let z_test = z_dist () in
  let k = ref 0. in
  Checkpointing.loop
    ~start_index:1
    ~end_index:batches
    ~var_stores:[ generator_vs; discriminator_vs ]
    ~checkpoint_base:"began.ot"
    ~checkpoint_every:(`seconds 600.)
    (fun ~index:batch_idx ->
      let learning_rate =
        learning_rate *. (0.95 **. (Float.of_int batch_idx /. lr_update_step))
      in
      Optimizer.set_learning_rate opt_d ~learning_rate;
      Optimizer.set_learning_rate opt_g ~learning_rate;
      let x_real =
        let index =
          Tensor.randint ~high:train_size ~size:[ batch_size ] ~options:(T Int64, Cpu)
        in
        Tensor.index_select images ~dim:0 ~index
        |> Tensor.to_type ~type_:(T Float)
        |> fun xs -> Tensor.((xs / f 127.5) - f 1.) |> Tensor.to_device ~device
      in
      let discriminator_loss, real_loss_d, fake_loss_d =
        Var_store.freeze generator_vs;
        Var_store.unfreeze discriminator_vs;
        Optimizer.zero_grad opt_d;
        let gen_z = Tensor.no_grad (fun () -> z_dist () |> Layer.forward generator) in
        let outputs_d_z = Layer.forward discriminator gen_z in
        let outputs_d_x = Layer.forward discriminator x_real in
        let real_loss_d = Tensor.(abs (outputs_d_x - x_real) |> mean) in
        let fake_loss_d = Tensor.(abs (outputs_d_z - gen_z) |> mean) in
        let k = Tensor.f !k |> Tensor.to_device ~device in
        let loss_d = Tensor.(real_loss_d - (k * fake_loss_d)) in
        Tensor.backward loss_d;
        Optimizer.step opt_d;
        loss_d, real_loss_d, fake_loss_d
      in
      let generator_loss =
        Var_store.unfreeze generator_vs;
        Var_store.freeze discriminator_vs;
        Optimizer.zero_grad opt_g;
        let gen_z = z_dist () |> Layer.forward generator in
        let outputs_g_z = Layer.forward discriminator gen_z in
        let loss_g = Tensor.(abs (outputs_g_z - gen_z) |> mean) in
        Tensor.backward loss_g;
        Optimizer.step opt_g;
        loss_g
      in
      let balance = Tensor.(float_value ((f gamma * real_loss_d) - fake_loss_d)) in
      k := Float.max 0. (Float.min 1. (!k +. (lambda_k *. balance)));
      if batch_idx % 100 = 0
      then
        Stdio.printf
          "batch %4d    d-loss: %12.6f    g-loss: %12.6f\n%!"
          batch_idx
          (Tensor.float_value discriminator_loss)
          (Tensor.float_value generator_loss);
      Caml.Gc.full_major ();
      if batch_idx % 25000 = 0 || (batch_idx < 100000 && batch_idx % 5000 = 0)
      then
        Tensor.no_grad (fun () -> Layer.forward generator z_test)
        |> Tensor.view ~size:[ batch_size; 3; img_size; img_size ]
        |> Tensor.to_device ~device:Cpu
        |> fun xs ->
        Tensor.((xs + f 1.) * f 127.5)
        |> Tensor.clamp ~min:(Scalar.float 0.) ~max:(Scalar.float 255.)
        |> Tensor.to_type ~type_:(T Uint8)
        |> write_samples ~filename:(Printf.sprintf "out%d.png" batch_idx))
