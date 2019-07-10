(* Transformer Network
   As introduced in "Attention Is All You Need", Vaswani et al. 2017.

   http://nlp.seas.harvard.edu/2018/04/03/attention.html
*)
open Base
open Torch

let layer_norm vs dim =
  let a = Var_store.new_var vs ~name:"a" ~shape:[ dim ] ~init:Ones in
  let b = Var_store.new_var vs ~name:"b" ~shape:[ dim ] ~init:Zeros in
  Layer.of_fn (fun xs ->
      let mu = Tensor.mean2 xs ~dim:[ -1 ] ~keepdim:true in
      let sigma = Tensor.std1 xs ~dim:[ -1 ] ~unbiased:true ~keepdim:true in
      Tensor.((a * (xs - mu) / (sigma + f 1e-6)) + b))

let sublayer_connection vs dim ~dropout =
  let layer_norm = layer_norm vs dim in
  fun xs ~sublayer ~is_training ->
    let ys =
      Layer.forward layer_norm xs
      |> fun xs -> sublayer xs ~is_training |> Tensor.dropout ~is_training ~p:dropout
    in
    Tensor.(xs + ys)

let attention ~query ~key ~value ~mask ~dropout ~is_training =
  let sqrt_d_k = Tensor.size query |> List.last_exn |> Float.of_int |> Float.sqrt in
  let scores = Tensor.matmul query (Tensor.transpose key ~dim0:(-2) ~dim1:(-1)) in
  let p_attn =
    let mask = Tensor.eq_scalar mask (Scalar.int 0) in
    Tensor.(scores / f sqrt_d_k)
    |> Tensor.masked_fill ~mask ~value:(Scalar.float (-1e9))
    |> Tensor.softmax ~dim:(-1)
    |> Tensor.dropout ~p:dropout ~is_training
  in
  Tensor.matmul p_attn value

let multi_headed_attention vs dim ~heads ~dropout =
  let linear_q = Layer.linear vs ~input_dim:dim dim in
  let linear_k = Layer.linear vs ~input_dim:dim dim in
  let linear_v = Layer.linear vs ~input_dim:dim dim in
  let linear_out = Layer.linear vs ~input_dim:dim dim in
  let d_k = dim / heads in
  fun ~query ~key ~value ~mask ~is_training ->
    let mask = Tensor.unsqueeze mask ~dim:1 in
    let batch_size = Tensor.size query |> List.hd_exn in
    let view_transpose xs =
      Tensor.view xs ~size:[ batch_size; -1; heads; d_k ]
      |> Tensor.transpose ~dim0:1 ~dim1:2
    in
    let query = Layer.forward linear_q query |> view_transpose in
    let key = Layer.forward linear_k key |> view_transpose in
    let value = Layer.forward linear_v value |> view_transpose in
    attention ~query ~key ~value ~mask ~dropout ~is_training
    |> Tensor.transpose ~dim0:1 ~dim1:2
    |> Tensor.contiguous
    |> Tensor.view ~size:[ batch_size; -1; heads * d_k ]
    |> Layer.forward linear_out

let positionwise_feed_forward vs dim ~dim_ff ~dropout =
  let linear_1 = Layer.linear vs ~input_dim:dim dim_ff in
  let linear_2 = Layer.linear vs ~input_dim:dim_ff dim in
  Layer.of_fn_ (fun xs ~is_training ->
      Layer.forward linear_1 xs
      |> Tensor.relu
      |> Tensor.dropout ~p:dropout ~is_training
      |> Layer.forward linear_2)

let encoder_layer vs dim ~heads ~dim_ff ~dropout =
  let attn = multi_headed_attention vs dim ~heads ~dropout in
  let feed_forward = positionwise_feed_forward vs dim ~dim_ff ~dropout in
  let sl1 = sublayer_connection vs dim ~dropout in
  let sl2 = sublayer_connection vs dim ~dropout in
  fun xs mask ~is_training ->
    sl1 xs ~sublayer:(fun xs -> attn ~query:xs ~key:xs ~value:xs ~mask) ~is_training
    |> sl2 ~sublayer:(Layer.forward_ feed_forward) ~is_training

let encoder vs dim ~n ~heads ~dim_ff ~dropout =
  let norm = layer_norm vs dim in
  let layers =
    List.init n ~f:(fun _index -> encoder_layer vs dim ~heads ~dim_ff ~dropout)
  in
  fun xs ~mask ~is_training ->
    List.fold layers ~init:xs ~f:(fun acc layer -> layer acc mask ~is_training)
    |> Layer.forward norm

let decoder_layer vs dim ~heads ~dim_ff ~dropout =
  let attn1 = multi_headed_attention vs dim ~heads ~dropout in
  let attn2 = multi_headed_attention vs dim ~heads ~dropout in
  let feed_forward = positionwise_feed_forward vs dim ~dim_ff ~dropout in
  let sl1 = sublayer_connection vs dim ~dropout in
  let sl2 = sublayer_connection vs dim ~dropout in
  let sl3 = sublayer_connection vs dim ~dropout in
  fun xs ~mem ~tgt_mask ~src_mask ~is_training ->
    sl1
      xs
      ~sublayer:(fun xs -> attn1 ~query:xs ~key:xs ~value:xs ~mask:tgt_mask)
      ~is_training
    |> sl2
         ~sublayer:(fun xs -> attn2 ~query:xs ~key:mem ~value:mem ~mask:src_mask)
         ~is_training
    |> sl3 ~sublayer:(Layer.forward_ feed_forward) ~is_training

let decoder vs dim ~n ~heads ~dim_ff ~dropout =
  let norm = layer_norm vs dim in
  let layers =
    List.init n ~f:(fun _index -> decoder_layer vs dim ~heads ~dim_ff ~dropout)
  in
  fun xs ~mem ~src_mask ~tgt_mask ~is_training ->
    List.fold layers ~init:xs ~f:(fun acc layer ->
        layer acc ~mem ~src_mask ~tgt_mask ~is_training)
    |> Layer.forward norm

let positional_encoding ?(max_len = 5000) vs dim ~dropout =
  let position = Tensor.arange ~end_:(Scalar.i max_len) ~options:(T Float, Cpu) in
  let div_term i =
    Float.exp (-2. *. Float.of_int i *. Float.log 10000.0 /. Float.of_int dim)
  in
  let sin i = Tensor.(sin (position * f (div_term i))) in
  let cos i = Tensor.(cos (position * f (div_term i))) in
  let pe =
    List.init dim ~f:(fun i -> if i % 2 = 0 then sin (i / 2) else cos (i / 2))
    |> Tensor.stack ~dim:1
    |> Tensor.unsqueeze ~dim:0
    |> Tensor.to_device ~device:(Var_store.device vs)
    |> Tensor.detach
  in
  Layer.of_fn_ (fun xs ~is_training ->
      let _, sz, _ = Tensor.shape3_exn xs in
      Tensor.(xs + Tensor.narrow pe ~dim:1 ~start:0 ~length:sz)
      |> Tensor.dropout ~p:dropout ~is_training)

let make_model vs ~src_vocab ~tgt_vocab ~n ~dim_ff ~dim_model ~heads ~dropout =
  let position = positional_encoding vs dim_model ~dropout in
  let encoder = encoder vs dim_model ~n ~heads ~dim_ff ~dropout in
  let decoder = decoder vs dim_model ~n ~heads ~dim_ff ~dropout in
  let src_e = Layer.embeddings vs ~num_embeddings:dim_model ~embedding_dim:src_vocab in
  let tgt_e = Layer.embeddings vs ~num_embeddings:dim_model ~embedding_dim:tgt_vocab in
  fun ~src ~tgt ~src_mask ~tgt_mask ~is_training ->
    let mem =
      Layer.forward src_e src
      |> Layer.forward_ position ~is_training
      |> encoder ~mask:src_mask ~is_training
    in
    Layer.forward tgt_e tgt
    |> Layer.forward_ position ~is_training
    |> decoder ~mem ~src_mask ~tgt_mask ~is_training

let () =
  (* Simple copy task. *)
  let device = Device.cuda_if_available () in
  let vs = Var_store.create ~name:"transformer" ~device () in
  let _model =
    make_model
      vs
      ~src_vocab:11
      ~tgt_vocab:11
      ~n:6
      ~dim_model:512
      ~dim_ff:2048
      ~heads:8
      ~dropout:0.1
  in
  let _optimizer = Optimizer.adam vs ~learning_rate:0.0001 in
  ()
