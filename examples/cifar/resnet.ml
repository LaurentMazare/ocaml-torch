(* CNN model for the CIFAR-10 dataset.

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   This reaches ~60% accuracy.
*)
open Base
open Torch
module Vs = Layer.Var_store

let batch_size = 64
let epochs = 5000
let learning_rate = 1e-4
let keep_probability = 1.0

let conv2d = Layer.conv2d_ ~padding:1 ~ksize:3 ~w_init:0.01 ~use_bias:false

let basic_block vs ~stride ~input_dim output_dim =
  let conv2d1 = conv2d vs ~stride ~input_dim output_dim in
  let conv2d2 = conv2d vs ~stride:1 ~input_dim:output_dim output_dim in
  fun xs ~is_training ->
    let shortcut =
      if stride = 1
      then xs
      else
        let xs = Tensor.avg_pool2d ~stride:(stride, stride) ~ksize:(stride, stride) xs in
        let zero_dims =
          match Tensor.shape xs with
          | [ batch_dim; channel_dim; w_dim; h_dim ] ->
            [ batch_dim; output_dim - channel_dim; w_dim; h_dim ]
          | _ -> assert false
        in
        Tensor.cat [xs; Tensor.zeros zero_dims ~device:(Vs.device vs)] ~dim:1
    in
    Layer.apply conv2d1 xs
    |> Tensor.dropout ~keep_probability ~is_training
    (* TODO: add some batch-norm or some group-norm. *)
    |> Tensor.relu
    |> Layer.apply conv2d2
    |> Tensor.dropout ~keep_probability ~is_training
    (* TODO: add some batch-norm or some group-norm. *)
    (* No final relu as per http://torch.ch/blog/2016/02/04/resnets.html *)
    |> fun xs -> Tensor.(xs + shortcut)

let block_stack vs ~stride ~depth ~input_dim output_dim =
  let basic_blocks =
    List.init depth ~f:(fun i ->
      basic_block vs output_dim
        ~stride:(if i = 0 then stride else 1)
        ~input_dim:(if i = 0 then input_dim else output_dim))
  in
  fun (xs : Tensor.t) ~is_training ->
    List.fold basic_blocks ~init:xs
      ~f:(fun acc basic_block -> basic_block acc ~is_training)

let resnet vs =
  let conv2d1 = conv2d vs ~stride:1 ~input_dim:3 ~activation:Relu 16 in
  let stack1 = block_stack vs ~stride:1 ~depth:2 ~input_dim:16 16 in
  let stack2 = block_stack vs ~stride:2 ~depth:2 ~input_dim:16 32 in
  let stack3 = block_stack vs ~stride:2 ~depth:2 ~input_dim:32 64 in
  let linear = Layer.linear vs ~w_init:0.01 ~activation:Softmax ~input_dim:64 Cifar_helper.label_count in
  fun xs ~is_training ->
    Tensor.reshape xs ~dims:Cifar_helper. [ -1; image_c; image_w; image_h ]
    |> Layer.apply conv2d1
    |> stack1 ~is_training
    |> stack2 ~is_training
    |> stack3 ~is_training
    |> Tensor.avg_pool2d ~ksize:(7, 7)
    |> Tensor.reshape ~dims:[ -1; 64 ]
    |> Layer.apply linear

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Some Torch_core.Device.Cuda
    end else None
  in
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Vs.create ~name:"resnet" ?device () in
  let model = resnet vs in
  let adam = Optimizer.adam (Vs.vars vs) ~learning_rate in
  let train_model = model ~is_training:true in
  let test_model = model ~is_training:false in
  Checkpointing.loop
    ~start_index:1 ~end_index:epochs
    ~named_tensors:(Vs.vars vs |> List.mapi ~f:(fun i t -> Int.to_string i, t))
    ~checkpoint_base:"resnet-ckpt"
    ~checkpoint_every:(`seconds 600.)
    (fun ~index:batch_idx ->
      let batch_images, batch_labels =
        Dataset_helper.train_batch cifar ?device ~batch_size ~batch_idx
      in
      (* Compute the cross-entropy loss. *)
      let loss = Tensor.(mean (- batch_labels * log (train_model batch_images +f 1e-6))) in

      Optimizer.backward_step adam ~loss;

      if batch_idx % 500 = 0 then begin
        (* Compute the validation error. *)
        let test_accuracy =
          Dataset_helper.batch_accuracy cifar `test ?device ~batch_size ~predict:test_model
        in
        Stdio.printf "%d %f %.2f%%\n%!" batch_idx (Tensor.float_value loss) (100. *. test_accuracy);
      end;
      Caml.Gc.full_major ())
