(* Training various models on the CIFAR-10 dataset.

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   The resnet model reaches ~92.5% accuracy.
*)
open Base
open Torch

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Cuda.set_benchmark_cudnn true;
      Torch_core.Device.Cuda
    end else Torch_core.Device.Cpu
  in
  let cifar = Cifar_helper.read_files ~with_caching:true () in
  let vs = Var_store.create ~name:"vs" ~device () in
  let { Model.model; epochs; lr_schedule; model_name; batch_size } =
    match Sys.argv with
    | [| _; "densenet" |] -> Densenet.model vs
    | [| _; "resnet" |] -> Resnet.model vs
    | [| _; "fast-resnet" |] -> Fast_resnet.model vs
    | [| _; "preact-resnet" |] | _ -> Preact_resnet.model vs
  in
  let sgd =
    Optimizer.sgd vs ~learning_rate:0.  ~momentum:0.9 ~weight_decay:5e-4 ~nesterov:true
  in
  let train_model xs = Layer.apply_ model xs ~is_training:true in
  let test_model xs = Layer.apply_ model xs ~is_training:false in
  let batches_per_epoch = Dataset_helper.batches_per_epoch cifar ~batch_size in
  Stdio.printf "Training %s for %d epochs.\n%!" model_name epochs;
  Checkpointing.loop ~start_index:1 ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:(model_name ^ ".ot")
    ~checkpoint_every:(`iters 25)
    (fun ~index:epoch_idx ->
      let start_time = Unix.gettimeofday () in
      let sum_loss = ref 0. in
      Dataset_helper.iter cifar ~device ~batch_size
        ~augmentation:[`flip; `crop_with_pad 4; `cutout 8]
        ~f:(fun batch_idx ~batch_images ~batch_labels ->
          Optimizer.set_learning_rate sgd
            ~learning_rate:(lr_schedule ~batch_idx ~batches_per_epoch ~epoch_idx);
          Optimizer.zero_grad sgd;
          let predicted = train_model batch_images in
          (* Compute the cross-entropy loss. *)
          let loss = Tensor.cross_entropy_for_logits predicted ~targets:batch_labels in
          sum_loss := !sum_loss +. Tensor.float_value loss;
          Stdio.printf "%d/%d %f\r%!"
            (1 + batch_idx)
            batches_per_epoch
            (!sum_loss /. Float.of_int (1 + batch_idx));
          Tensor.backward loss;
          Optimizer.step sgd);

      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy cifar `test ~device ~batch_size ~predict:test_model
      in
      Stdio.printf "%d %.0fs %f %.2f%%\n%!"
        epoch_idx
        (Unix.gettimeofday () -. start_time)
        (!sum_loss /. Float.of_int batches_per_epoch)
        (100. *. test_accuracy);
    )
