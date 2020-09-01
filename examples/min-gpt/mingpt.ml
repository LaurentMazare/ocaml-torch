(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/minGPT
*)
open Base
open Torch

let learning_rate = 0.0003
let batch_size = 64
let seq_len = 128
let epochs = 100
let sampling_length = 2048
let temperature = 1.0

type config =
  { vocab_size : int
  ; n_embd : int
  ; n_head : int
  ; n_layer : int
  ; block_size : int
  ; attn_pdrop : float
  ; resid_pdrop : float
  ; embd_pdrop : float
  }

let causal_self_attention vs cfg =
  let linear n = Layer.linear Var_store.(vs / n) ~input_dim:cfg.n_embd cfg.n_embd in
  let key = linear "key" in
  let query = linear "query" in
  let value = linear "value" in
  let proj = linear "proj" in
  let mask_init =
    Tensor.ones [ cfg.block_size; cfg.block_size ] ~device:(Var_store.device vs)
    |> Tensor.tril ~diagonal:0
    |> Tensor.view ~size:[ 1; 1; cfg.block_size; cfg.block_size ]
  in
  (* TODO: train mask *)
  let mask = Tensor.eq_scalar mask_init (Scalar.float 0.) in
  Layer.of_fn_ (fun xs ~is_training ->
      let sz_b, sz_t, sz_c = Tensor.shape3_exn xs in
      let sz_c_over_n_head = sz_c / cfg.n_head in
      let apply_linear l =
        let size = [ sz_b; sz_t; cfg.n_head; sz_c_over_n_head ] in
        Layer.forward l xs |> Tensor.view ~size |> Tensor.transpose ~dim0:1 ~dim1:2
      in
      let k = apply_linear key in
      let q = apply_linear query in
      let v = apply_linear value in
      let att = Tensor.matmul q (Tensor.transpose k ~dim0:(-2) ~dim1:(-1)) in
      let att =
        Tensor.(att / f (Float.of_int sz_c_over_n_head |> Float.sqrt))
        |> Tensor.masked_fill ~mask ~value:(Scalar.f Float.neg_infinity)
        |> Tensor.softmax ~dim:(-1) ~dtype:(T Float)
        |> Tensor.dropout ~p:cfg.attn_pdrop ~is_training
      in
      Tensor.matmul att v
      |> Tensor.transpose ~dim0:1 ~dim1:2
      |> Tensor.contiguous
      |> Tensor.view ~size:[ sz_b; sz_t; sz_c ]
      |> Layer.forward proj
      |> Tensor.dropout ~p:cfg.resid_pdrop ~is_training)

let block vs cfg =
  let { n_embd; resid_pdrop; _ } = cfg in
  let ln1 = Layer.layer_norm Var_store.(vs / "ln1") n_embd in
  let ln2 = Layer.layer_norm Var_store.(vs / "ln2") n_embd in
  let attn = causal_self_attention vs cfg in
  let lin1 = Layer.linear Var_store.(vs / "lin1") ~input_dim:n_embd (4 * n_embd) in
  let lin2 = Layer.linear Var_store.(vs / "lin2") ~input_dim:(4 * n_embd) n_embd in
  Layer.of_fn_ (fun xs ~is_training ->
      let xs =
        Tensor.( + ) xs (Layer.forward ln1 xs |> Layer.forward_ attn ~is_training)
      in
      let ys =
        Layer.forward ln2 xs
        |> Layer.forward lin1
        |> Tensor.gelu
        |> Layer.forward lin2
        |> Tensor.dropout ~p:resid_pdrop ~is_training
      in
      Tensor.(xs + ys))

let gpt vs cfg =
  let { vocab_size; n_embd; block_size; n_layer; _ } = cfg in
  let tok_emb =
    Layer.embeddings
      Var_store.(vs / "tok_emb")
      ~num_embeddings:vocab_size
      ~embedding_dim:n_embd
  in
  let pos_emb =
    Var_store.new_var vs ~shape:[ 1; block_size; n_embd ] ~name:"pos_emb" ~init:Zeros
  in
  let ln_f = Layer.layer_norm Var_store.(vs / "ln_f") n_embd in
  let head =
    Layer.linear Var_store.(vs / "head") ~input_dim:n_embd vocab_size ~use_bias:false
  in
  let blocks =
    List.init n_layer ~f:(fun i -> block Var_store.(vs // i) cfg) |> Layer.sequential_
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let _sz_b, sz_t = Tensor.shape2_exn xs in
      let tok_emb = Layer.forward tok_emb xs in
      let pos_emb = Tensor.narrow pos_emb ~dim:1 ~start:0 ~length:sz_t in
      Tensor.(tok_emb + pos_emb)
      |> Tensor.dropout ~p:cfg.embd_pdrop ~is_training
      |> Layer.forward_ blocks ~is_training
      |> Layer.forward ln_f
      |> Layer.forward head)

let sample cfg ~gpt ~dataset ~device =
  let { block_size; _ } = cfg in
  let input = Tensor.zeros [ 1; block_size ] ~kind:(T Int64) ~device in
  List.init sampling_length ~f:Fn.id
  |> List.fold_map ~init:input ~f:(fun input _idx ->
         Caml.Gc.full_major ();
         let logits =
           Layer.forward_ gpt input ~is_training:false |> Tensor.select ~dim:1 ~index:(-1)
         in
         let logits = Tensor.(logits / f temperature) in
         let sampled_y =
           Tensor.softmax logits ~dim:(-1) ~dtype:(T Float)
           |> Tensor.multinomial ~num_samples:1 ~replacement:true
         in
         let sampled_char =
           Text_helper.char dataset ~label:(Tensor.int_value sampled_y)
         in
         let input =
           Tensor.cat [ input; Tensor.view sampled_y ~size:[ 1; 1 ] ] ~dim:1
           |> Tensor.narrow ~dim:1 ~start:1 ~length:block_size
         in
         input, sampled_char)
  |> snd
  |> String.of_char_list

let train vs ~cfg ~gpt ~dataset =
  let device = Var_store.device vs in
  let labels = Text_helper.labels dataset in
  let adam = Optimizer.adam vs ~learning_rate in
  let batches_per_epoch = (Text_helper.total_length dataset - seq_len) / batch_size in
  Checkpointing.loop
    ~start_index:1
    ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"min-gpt.ot"
    ~checkpoint_every:(`iters 1)
    (fun ~index:epoch_idx ->
      Stdio.Out_channel.write_all
        (Printf.sprintf "out.txt.%d" epoch_idx)
        ~data:(sample cfg ~gpt ~dataset ~device);
      let start_time = Unix.gettimeofday () in
      let sum_loss = ref 0. in
      Text_helper.iter dataset ~device ~batch_size ~seq_len ~f:(fun batch_idx ~xs ~ys ->
          let logits = Layer.forward_ gpt xs ~is_training:true in
          (* Compute the cross-entropy loss. *)
          let loss =
            Tensor.cross_entropy_for_logits
              (Tensor.view logits ~size:[ batch_size * seq_len; labels ])
              ~targets:(Tensor.view ys ~size:[ batch_size * seq_len ])
          in
          sum_loss := !sum_loss +. Tensor.float_value loss;
          Stdio.printf
            "%d/%d %f\r%!"
            batch_idx
            batches_per_epoch
            (!sum_loss /. Float.of_int (1 + batch_idx));
          Optimizer.backward_step ~clip_grad:(Norm2 4.) adam ~loss);
      Stdio.printf
        "%d %.0fs %f\n%!"
        epoch_idx
        (Unix.gettimeofday () -. start_time)
        (!sum_loss /. Float.of_int batches_per_epoch))

let () =
  let device = Device.cuda_if_available () in
  let dataset = Text_helper.create ~filename:"data/input.txt" in
  let vs = Var_store.create ~name:"min-gpt" ~device () in
  let labels = Text_helper.labels dataset in
  Stdio.printf
    "Dataset loaded, length: %d, labels: %d.\n%!"
    (Text_helper.total_length dataset)
    labels;
  let cfg =
    { vocab_size = labels
    ; n_embd = 512
    ; n_head = 8
    ; n_layer = 8
    ; block_size = seq_len
    ; attn_pdrop = 0.1
    ; resid_pdrop = 0.1
    ; embd_pdrop = 0.1
    }
  in
  let gpt = gpt vs cfg in
  match Caml.Sys.argv with
  | [| _bin |] | [| _bin; "train" |] -> train vs ~gpt ~cfg ~dataset
  | [| _bin; "sample"; filename |] ->
    let named_tensors = Var_store.all_vars vs in
    Serialize.load_multi_ ~named_tensors ~filename;
    sample cfg ~gpt ~dataset ~device |> Stdio.print_endline
  | _ -> failwith "usage: mingpt (train|sample weight.ot)"
