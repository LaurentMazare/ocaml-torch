(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/char-rnn
*)
open Base
open Torch

let learning_rate = 0.04
let hidden_size = 256
let seq_len = 180
let batch_size = 128
let epochs = 100

let () =
  let device =
    if Cuda.is_available ()
    then begin
      Stdio.printf "Using cuda, devices: %d\n%!" (Cuda.device_count ());
      Torch_core.Device.Cuda
    end else Torch_core.Device.Cpu
  in
  let dataset = Text_helper.create ~filename:"data/input.txt" in
  let vs = Var_store.create ~name:"char-rnn" ~device () in
  let labels = Text_helper.labels dataset in
  Stdio.printf "Dataset loaded, length: %d, labels: %d.\n%!"
    (Text_helper.total_length dataset) labels;

  let lstm = Layer.Lstm.create vs ~input_dim:labels ~hidden_size in
  let linear = Layer.linear vs ~input_dim:hidden_size labels in

  let adam = Optimizer.adam vs ~learning_rate in
  let batches_per_epoch = (Text_helper.total_length dataset - seq_len) / batch_size in
  Checkpointing.loop ~start_index:1 ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"char-rnn.ot"
    ~checkpoint_every:(`iters 10)
    (fun ~index:epoch_idx ->
      let start_time = Unix.gettimeofday () in
      let sum_loss = ref 0. in
      Text_helper.iter dataset ~device ~batch_size ~seq_len ~f:(fun batch_idx ~xs ~ys ->
        Optimizer.zero_grad adam;
        let onehot =
          let xs = Tensor.view xs ~size:[ batch_size; seq_len; 1 ] in
          let one = Tensor.ones [] in
          Tensor.zeros [ batch_size; seq_len; labels ] ~device
          |> Tensor.scatter_ ~dim:2 ~src:one ~index:xs
        in
        let lstm_out, _ = Layer.Lstm.seq lstm onehot in
        let logits = Layer.apply linear lstm_out in
        (* Compute the cross-entropy loss. *)
        let loss =
          Tensor.cross_entropy_for_logits
            (Tensor.view logits ~size:[ batch_size * seq_len; labels ])
            ~targets:(Tensor.view ys ~size:[ batch_size * seq_len ])
        in
        sum_loss := !sum_loss +. Tensor.float_value loss;
        Stdio.printf "%d/%d %f\r%!"
          batch_idx batches_per_epoch
          (!sum_loss /. Float.of_int (1 + batch_idx));
          Tensor.backward loss;
        Optimizer.step adam);
      Stdio.printf "%d %.0fs %f\n%!"
        epoch_idx
        (Unix.gettimeofday () -. start_time)
        (!sum_loss /. Float.of_int batches_per_epoch)
    )
