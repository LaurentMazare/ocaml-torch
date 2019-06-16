(* Translation with a Sequence to Sequence Model and Attention.

   This follows the line of the PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

   The dataset can be downloaded from the following link:
   https://download.pytorch.org/tutorial/data.zip
   The eng-fra.txt file should be moved in the data directory.
*)
open! Base
open Torch

let encoder vs ~input_size ~hidden_size =
  let embedding =
    Layer.embeddings vs ~num_embeddings:input_size ~embedding_dim:hidden_size
  in
  let gru = Layer.Gru.create vs ~input_dim:hidden_size ~hidden_size in
  fun input_ hidden ->
    let hidden =
      Layer.forward embedding input_
      |> Tensor.view ~size:[ 1; -1 ]
      |> Layer.Gru.step gru hidden
    in
    hidden, hidden

(* Decoder without attention. *)
let decoder vs ~hidden_size ~output_size =
  let embedding =
    Layer.embeddings vs ~num_embeddings:output_size ~embedding_dim:hidden_size
  in
  let gru = Layer.Gru.create vs ~input_dim:hidden_size ~hidden_size in
  let linear = Layer.linear vs ~input_dim:hidden_size output_size in
  fun input_ hidden ->
    let hidden =
      Layer.forward embedding input_
      |> Tensor.view ~size:[ 1; -1 ]
      |> Tensor.relu
      |> Layer.Gru.step gru hidden
    in
    Layer.forward linear hidden |> Tensor.softmax ~dim:(-1), hidden

let () =
  let dataset = Dataset.create ~input_lang:"eng" ~output_lang:"fra" |> Dataset.reverse in
  let ilang = Dataset.input_lang dataset in
  let olang = Dataset.output_lang dataset in
  Stdio.printf "Input:  %s %d words.\n%!" (Lang.name ilang) (Lang.length ilang);
  Stdio.printf "Output: %s %d words.\n%!" (Lang.name olang) (Lang.length olang);
  let device = Device.cuda_if_available () in
  let vs = Var_store.create ~name:"seq2seq" ~device () in
  let encoder = encoder vs in
  let decoder = decoder vs in
  ignore (encoder, decoder)
