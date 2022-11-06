(* Inference code from Progressive Growing of GANs.

   This is based on the PyTorch implementation:
     https://github.com/ptrblck/prog_gans_pytorch_inference
   Which itself is based on the Theano/Lasagne implementation of
     Progressive Growing of GANs paper from tkarras:
     https://github.com/tkarras/progressive_growing_of_gans
*)
open Base
open Torch

let leaky_relu xs = Tensor.(max xs (xs * f 0.2))

let pixel_norm xs =
  Tensor.(
    xs / (sqrt (mean_dim (xs * xs) ~dim:(Some [ 1 ]) ~keepdim:true ~dtype:(T Float)) + f 1e-8))

let w_scale_layer vs ~size:sz =
  let vs = Var_store.sub vs "wscale" in
  let scale_ = Var_store.new_var vs ~shape:[ 1 ] ~init:Zeros ~name:"scale" in
  let b = Var_store.new_var vs ~shape:[ sz ] ~init:Zeros ~name:"b" in
  Layer.of_fn (fun xs ->
      let b = Tensor.view b ~size:[ 1; sz; 1; 1 ] in
      let x0, _, x2, x3 = Tensor.shape4_exn xs in
      Tensor.((xs * scale_) + expand b ~implicit:false ~size:[ x0; sz; x2; x3 ]))

let norm_conv_block ~vs ~ksize ~padding ~input_dim dim =
  let conv =
    Layer.conv2d_
      (Var_store.sub vs "conv")
      ~ksize
      ~stride:1
      ~padding
      ~use_bias:false
      ~input_dim
      dim
  in
  let wscale = w_scale_layer vs ~size:dim in
  Layer.of_fn (fun xs ->
      pixel_norm xs |> Layer.forward conv |> Layer.forward wscale |> leaky_relu)

let norm_upscale_conv_block ~vs ~ksize ~padding ~input_dim dim =
  let conv =
    Layer.conv2d_
      (Var_store.sub vs "conv")
      ~ksize
      ~stride:1
      ~padding
      ~use_bias:false
      ~input_dim
      dim
  in
  let wscale = w_scale_layer vs ~size:dim in
  Layer.of_fn (fun xs ->
      let _, _, h, w = Tensor.shape4_exn xs in
      pixel_norm xs
      |> Tensor.upsample_nearest2d
           ~output_size:[ 2 * h; 2 * w ]
           ~scales_h:(Some 2.0)
           ~scales_w:(Some 2.0)
      |> Layer.forward conv
      |> Layer.forward wscale
      |> leaky_relu)

let create_generator vs =
  let features_vs = Var_store.sub vs "features" in
  let features =
    List.mapi
      ~f:(fun i l -> l ~vs:Var_store.(features_vs / Int.to_string i))
      [ norm_conv_block ~ksize:4 ~padding:3 ~input_dim:512 512
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:512 512
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:512 256
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:256 256
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:256 128
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:128 128
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:128 64
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:64 64
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:64 32
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:32 32
      ; norm_upscale_conv_block ~ksize:3 ~padding:1 ~input_dim:32 16
      ; norm_conv_block ~ksize:3 ~padding:1 ~input_dim:16 16
      ]
    |> Layer.sequential
  in
  let conv =
    Layer.conv2d_
      Var_store.(vs / "output" / "conv")
      ~ksize:1
      ~stride:1
      ~padding:0
      ~use_bias:false
      ~input_dim:16
      3
  in
  let wscale = w_scale_layer Var_store.(vs / "output") ~size:3 in
  Layer.of_fn (fun xs ->
      Layer.forward features xs
      |> pixel_norm
      |> Layer.forward conv
      |> Layer.forward wscale)

let () =
  let module Sys = Caml.Sys in
  if Array.length Sys.argv <> 2
  then Printf.failwithf "usage: %s prog-gan.ot" Sys.argv.(0) ();
  Torch_core.Wrapper.manual_seed 42;
  let vs = Var_store.create ~name:"vs" () in
  let generator = create_generator vs in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  for i = 1 to 50 do
    let image = Tensor.(randn [ 1; 512; 1; 1 ]) |> Layer.forward generator in
    let image = Tensor.(image - minimum image) in
    let image = Tensor.(image / maximum image * f 255.) in
    Torch_vision.Image.write_image image ~filename:(Printf.sprintf "out%d.png" i);
    Caml.Gc.full_major ()
  done
