(* GAN stability
   adapted from https://github.com/LMescheder/GAN_stability/
*)
open Base
open Torch

let img_size = 128
let latent_dim = 128
let batch_size = 16
let learning_rate = 1e-4
let reg_param = 10.
let nf = 32

let batches = 10**8

let leaky_relu xs = Tensor.(max xs (xs * f 0.2))
let conv2d = Layer.conv2d_ ~stride:1
let upsample xs =
  let _, _, x, y = Tensor.shape4_exn xs in
  Tensor.upsample_nearest2d xs ~output_size:[ 2*x; 2*y ]
let avg_pool2d =
  Tensor.avg_pool2d ~ksize:(3, 3) ~stride:(2, 2) ~padding:(1, 1)

(* Use the resnet2 model similar to:
   https://github.com/LMescheder/GAN_stability/blob/master/gan_training/models/resnet2.py
*)
let resnet_block vs ~input_dim output_dim =
  let hidden_dim = Int.min input_dim output_dim in
  let c0 = conv2d vs ~ksize:3 ~padding:1 ~input_dim hidden_dim in
  let c1 = conv2d vs ~ksize:3 ~padding:1 ~input_dim:hidden_dim output_dim in
  let shortcut =
    if input_dim = output_dim
    then Layer.id
    else conv2d vs ~ksize:1 ~padding:0 ~use_bias:false ~input_dim output_dim
  in
  Layer.of_fn (fun xs ->
      leaky_relu xs
      |> Layer.apply c0
      |> leaky_relu
      |> Layer.apply c1
      |> fun ys -> Tensor.(Layer.apply shortcut xs + ys * f 0.1))

let create_generator vs =
  let s0 = img_size / 32 in
  let fc = Layer.linear vs ~input_dim:latent_dim (16 * nf * s0 * s0) in
  let rn00 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn01 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn10 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn11 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn20 = resnet_block vs ~input_dim:(16*nf) ( 8*nf) in
  let rn21 = resnet_block vs ~input_dim:( 8*nf) ( 8*nf) in
  let rn30 = resnet_block vs ~input_dim:( 8*nf) ( 4*nf) in
  let rn31 = resnet_block vs ~input_dim:( 4*nf) ( 4*nf) in
  let rn40 = resnet_block vs ~input_dim:( 4*nf) ( 2*nf) in
  let rn41 = resnet_block vs ~input_dim:( 2*nf) ( 2*nf) in
  let rn50 = resnet_block vs ~input_dim:( 2*nf) ( 1*nf) in
  let rn51 = resnet_block vs ~input_dim:( 1*nf) ( 1*nf) in
  let conv = conv2d vs ~ksize:3 ~padding:1 ~input_dim:nf 3 in
  fun rand_input ->
    Tensor.to_device rand_input ~device:(Var_store.device vs)
    |> Layer.apply fc
    |> Tensor.view ~size:[ batch_size; 16*nf; s0; s0 ]
    |> Layer.apply rn00
    |> Layer.apply rn01
    |> upsample
    |> Layer.apply rn10
    |> Layer.apply rn11
    |> upsample
    |> Layer.apply rn20
    |> Layer.apply rn21
    |> upsample
    |> Layer.apply rn30
    |> Layer.apply rn31
    |> upsample
    |> Layer.apply rn40
    |> Layer.apply rn41
    |> upsample
    |> Layer.apply rn50
    |> Layer.apply rn51
    |> leaky_relu
    |> Layer.apply conv
    |> Tensor.tanh

let create_discriminator vs =
  let s0 = img_size / 32 in
  let conv = conv2d vs ~ksize:3 ~padding:1 ~input_dim:3 nf in
  let rn00 = resnet_block vs ~input_dim:( 1*nf) ( 1*nf) in
  let rn01 = resnet_block vs ~input_dim:( 1*nf) ( 2*nf) in
  let rn10 = resnet_block vs ~input_dim:( 2*nf) ( 2*nf) in
  let rn11 = resnet_block vs ~input_dim:( 2*nf) ( 4*nf) in
  let rn20 = resnet_block vs ~input_dim:( 4*nf) ( 4*nf) in
  let rn21 = resnet_block vs ~input_dim:( 4*nf) ( 8*nf) in
  let rn30 = resnet_block vs ~input_dim:( 8*nf) ( 8*nf) in
  let rn31 = resnet_block vs ~input_dim:( 8*nf) (16*nf) in
  let rn40 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn41 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn50 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let rn51 = resnet_block vs ~input_dim:(16*nf) (16*nf) in
  let fc = Layer.linear vs ~input_dim:(16 * nf * s0 * s0) 1 in
  fun xs ->
    Tensor.to_device xs ~device:(Var_store.device vs)
    |> Layer.apply conv
    |> Layer.apply rn00
    |> Layer.apply rn01
    |> avg_pool2d
    |> Layer.apply rn10
    |> Layer.apply rn11
    |> avg_pool2d
    |> Layer.apply rn20
    |> Layer.apply rn21
    |> avg_pool2d
    |> Layer.apply rn30
    |> Layer.apply rn31
    |> avg_pool2d
    |> Layer.apply rn40
    |> Layer.apply rn41
    |> avg_pool2d
    |> Layer.apply rn50
    |> Layer.apply rn51
    |> Tensor.view ~size:[ batch_size; 16*nf*s0*s0 ]
    |> leaky_relu
    |> Layer.apply fc

let z_dist () = Tensor.randn [ batch_size; latent_dim ]

let write_samples samples ~filename =
  List.init 4 ~f:(fun i ->
      List.init 4 ~f:(fun j ->
          Tensor.narrow samples ~dim:0 ~start:(4*i + j) ~length:1)
      |> Tensor.cat ~dim:2)
  |> Tensor.cat ~dim:3
  |> Torch_vision.Image.write_image ~filename

let grad2 d_out x_in =
  let grad_dout =
    Tensor.run_backward [ Tensor.sum d_out ] [ x_in ]
      ~create_graph:true ~keep_graph:true
    |> List.hd_exn
  in
  Tensor.(grad_dout * grad_dout)
  |> Tensor.view ~size:[ batch_size; -1 ]
  |> Tensor.sum4 ~dim:[1] ~keepdim:false

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Cuda.set_benchmark_cudnn true;
      Torch_core.Device.Cuda
    end else Torch_core.Device.Cpu
  in

  if Array.length Sys.argv < 2
  then Printf.failwithf "Usage: %s images.ot" Sys.argv.(0) ();

  let bce_loss_with_logits ys ~target =
    Tensor.bce_loss (Tensor.sigmoid ys) ~targets:Tensor.(ones_like1 ys * f target)
  in

  let images = Serialize.load ~filename:Sys.argv.(1) in
  let train_size = Tensor.shape images |> List.hd_exn in

  let generator_vs = Var_store.create ~name:"gen" ~device () in
  let generator = create_generator generator_vs in
  let opt_g = Optimizer.adam generator_vs ~learning_rate in

  let discriminator_vs = Var_store.create ~name:"disc" ~device () in
  let discriminator = create_discriminator discriminator_vs in
  let opt_d = Optimizer.adam discriminator_vs ~learning_rate in

  let z_test = z_dist () in
  Checkpointing.loop ~start_index:1 ~end_index:batches
    ~var_stores:[ generator_vs; discriminator_vs ]
    ~checkpoint_base:"gan-stability.ot"
    ~checkpoint_every:(`seconds 600.)
    (fun ~index:batch_idx ->
       let x_real =
         let index = Tensor.randint1 ~high:train_size ~size:[ batch_size ] ~options:(Int64, Cpu) in
         Tensor.index_select images ~dim:0 ~index
         |> Tensor.to_type ~type_:Float
         |> fun xs -> Tensor.(xs / f 127.5 - f 1.)
       in
       let discriminator_loss =
         Var_store.freeze generator_vs;
         Var_store.unfreeze discriminator_vs;
         Optimizer.zero_grad opt_d;
         let x_real = Tensor.set_requires_grad x_real ~r:true in
         let d_real = discriminator x_real in
         let d_loss_real = bce_loss_with_logits d_real ~target:1. in
         Tensor.backward d_loss_real ~keep_graph:true;
         let reg = Tensor.(f reg_param * grad2 d_real x_real |> mean) in
         Tensor.backward reg;
         let x_fake = Tensor.no_grad (fun () -> z_dist () |> generator) in
         let x_fake = Tensor.set_requires_grad x_fake ~r:true in
         let d_fake = discriminator x_fake in
         let d_loss_fake = bce_loss_with_logits d_fake ~target:0. in
         Tensor.backward d_loss_fake;
         Optimizer.step opt_d;
         Tensor.(+) d_loss_real d_loss_fake
       in
       let generator_loss =
         Var_store.unfreeze generator_vs;
         Var_store.freeze discriminator_vs;
         Optimizer.zero_grad opt_g;
         let z = z_dist () in
         let x_fake = generator z in
         let d_fake = discriminator x_fake in
         let g_loss = bce_loss_with_logits d_fake ~target:1. in
         Tensor.backward g_loss;
         Optimizer.step opt_g;
         g_loss
       in
       if batch_idx % 100 = 0
       then
         Stdio.printf "batch %4d    d-loss: %12.6f    g-loss: %12.6f\n%!"
           batch_idx
           (Tensor.float_value discriminator_loss)
           (Tensor.float_value generator_loss);
       Caml.Gc.full_major ();
       if batch_idx % 25000 = 0 || (batch_idx < 100000 && batch_idx % 5000 = 0)
       then
         Tensor.no_grad (fun () -> generator z_test)
         |> Tensor.view ~size:[ batch_size; 3; img_size; img_size ]
         |> Tensor.transpose ~dim0:2 ~dim1:3
         |> Tensor.to_device ~device:Cpu
         |> fun xs -> Tensor.((xs + f 1.) * f 127.5)
         |> Tensor.clamp_ ~min:(Scalar.float 0.) ~max:(Scalar.float 255.)
         |> Tensor.to_type ~type_:Uint8
         |> write_samples ~filename:(Printf.sprintf "out%d.png" batch_idx)
    )
