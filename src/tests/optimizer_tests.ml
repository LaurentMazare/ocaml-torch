open Base
open Torch

let square x = Tensor.(x * x)

let%expect_test _ =
  Torch_core.Wrapper.manual_seed 42;
  let xs =
    List.init 15 ~f:Float.of_int |> Tensor.float_vec |> Tensor.view ~size:[ 15; 1 ]
  in
  let ys = Tensor.((xs * f 0.42) + f 1.337) in
  (* Build a linear model and fit it to the data. *)
  let vs = Var_store.create ~name:"vs" () in
  let opt = Optimizer.sgd vs ~learning_rate:1e-3 in
  let linear = Layer.linear vs ~input_dim:1 1 in
  for index = 1 to 100 do
    Optimizer.zero_grad opt;
    let ys_ = Layer.forward linear xs in
    let loss = Tensor.(mean (square (ys - ys_))) in
    if index % 10 = 0
    then Stdio.printf !"%d %{sexp:float}\n" index (Tensor.to_float0_exn loss);
    Tensor.backward loss;
    Optimizer.step opt
  done;
  let ys_ = Layer.forward linear xs in
  let loss = Tensor.(mean ((ys - ys_) * (ys - ys_))) in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn loss);
  [%expect
    {|
        10 0.48697763681411743
        20 0.099454931914806366
        30 0.07820592075586319
        40 0.076292321085929871
        50 0.075407594442367554
        60 0.074585653841495514
        70 0.073775477707386017
        80 0.072974249720573425
        90 0.07218170166015625
        100 0.071397751569747925
        0.0713198333978653
      |}]
