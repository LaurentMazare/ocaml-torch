open Base
open Sexplib.Conv
open Torch

let batch_size = 5
let seq_len = 3
let input_dim = 2
let output_dim = 4

let%expect_test _ =
  let vs = Var_store.create ~name:"test-vs" () in
  let gru = Layer.Gru.create vs ~input_dim ~hidden_size:output_dim in
  let input = Tensor.randn [ batch_size; input_dim ] in
  let (`state state) = Layer.Gru.step gru (Layer.Gru.zero_state gru ~batch_size) input in
  Stdio.printf !"%{sexp:int list}\n" (Tensor.shape state);
  [%expect {| (5 4) |}];
  let input = Tensor.randn [ batch_size; seq_len; input_dim ] in
  let out, _ = Layer.Gru.seq gru input in
  Stdio.printf !"%{sexp:int list}\n" (Tensor.shape out);
  [%expect {| (5 3 4) |}]

let%expect_test _ =
  let vs = Var_store.create ~name:"test-vs" () in
  let lstm = Layer.Lstm.create vs ~input_dim ~hidden_size:output_dim in
  let input = Tensor.randn [ batch_size; input_dim ] in
  let (`h_c (h, c)) =
    Layer.Lstm.step lstm (Layer.Lstm.zero_state lstm ~batch_size) input
  in
  Stdio.printf !"%{sexp:int list}\n" (Tensor.shape h);
  Stdio.printf !"%{sexp:int list}\n" (Tensor.shape c);
  [%expect {|
    (5 4)
    (5 4) |}];
  let input = Tensor.randn [ batch_size; seq_len; input_dim ] in
  let out, _ = Layer.Lstm.seq lstm input in
  Stdio.printf !"%{sexp:int list}\n" (Tensor.shape out);
  [%expect {| (5 3 4) |}]
