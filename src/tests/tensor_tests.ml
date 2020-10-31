open Base
open Sexplib.Conv
open Torch

let%expect_test _ =
  let t = Tensor.(f 41. + f 1.) in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
  [%expect {|
        42
      |}];
  let t = Tensor.float_vec [ 1.; 42.; 1337. ] in
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t * t));
  [%expect {|
        (1 1764 1787569)
      |}];
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t + f 1.5));
  [%expect {|
        (2.5 43.5 1338.5)
      |}]

let%expect_test _ =
  let open Tensor in
  let t = zeros [ 4; 2 ] in
  t.%.{[ 1; 1 ]} <- 42.0;
  t.%.{[ 3; 0 ]} <- 1.337;
  for i = 0 to 3 do
    Stdio.printf "%f %f\n" t.%.{[ i; 0 ]} t.%.{[ i; 1 ]}
  done;
  [%expect
    {|
        0.000000 0.000000
        0.000000 42.000000
        0.000000 0.000000
        1.337000 0.000000
      |}]

let%expect_test _ =
  let open Tensor in
  let t = zeros [ 5; 2 ] in
  t += f 1.;
  narrow t ~dim:0 ~start:1 ~length:3 += f 2.;
  narrow t ~dim:1 ~start:1 ~length:1 -= f 3.;
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn t);
  [%expect {| ((1 -2) (3 0) (3 0) (3 0) (1 -2)) |}]

let%expect_test _ =
  let t = List.init 5 ~f:Float.of_int |> Tensor.float_vec in
  let array = Tensor.to_float1_exn t in
  let array_narrow = Tensor.narrow t ~dim:0 ~start:1 ~length:3 |> Tensor.to_float1_exn in
  Stdio.printf !"%{sexp:float array}\n" array;
  Stdio.printf !"%{sexp:float array}\n" array_narrow;
  [%expect {|
    (0 1 2 3 4)
    (1 2 3) |}]

let%expect_test _ =
  let t = Tensor.of_int2 [| [| 3; 4; 5 |]; [| 2; 3; 4 |] |] in
  Tensor.(narrow t ~dim:1 ~start:0 ~length:1 += of_int0 42);
  Stdio.printf !"%{sexp:int array array}\n" (Tensor.to_int2_exn t);
  [%expect {| ((45 4 5) (44 3 4)) |}]

let%expect_test _ =
  let t = Tensor.zeros [ 2; 3; 2 ] in
  let u = Tensor.narrow t ~dim:1 ~start:1 ~length:2 in
  let v = Tensor.get u 1 in
  let w = Tensor.copy v in
  Tensor.(w += f 1.);
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn w);
  [%expect {| ((1 1) (1 1)) |}];
  Stdio.printf !"%{sexp:float array array array}\n" (Tensor.to_float3_exn t);
  [%expect {| (((0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0))) |}];
  Tensor.(v += f 1.);
  Stdio.printf !"%{sexp:float array array array}\n" (Tensor.to_float3_exn t);
  [%expect {| (((0 0) (0 0) (0 0)) ((0 0) (1 1) (1 1))) |}]

let%expect_test _ =
  let logits = Tensor.of_float1 [| -1.; 0.5; 0.25; 0.; 2.; 4.; -1. |] in
  let eval_and_print ~target =
    let bce1 =
      Tensor.(bce_loss (sigmoid logits) ~targets:(ones_like logits * f target))
    in
    let bce2 =
      Tensor.(bce_loss_with_logits logits ~targets:(ones_like logits * f target))
    in
    let bce3 =
      Tensor.(
        (-f target * log (sigmoid logits))
        - ((f 1. - f target) * log (f 1. - sigmoid logits)))
      |> Tensor.mean
    in
    Stdio.printf
      !"%{sexp:float} %{sexp:float} %{sexp:float}\n"
      (Tensor.to_float0_exn bce1)
      (Tensor.to_float0_exn bce2)
      (Tensor.to_float0_exn bce3)
  in
  eval_and_print ~target:0.;
  [%expect {| 1.3235375881195068 1.3235378265380859 1.3235375881195068 |}];
  eval_and_print ~target:0.5;
  [%expect {| 0.98425191640853882 0.98425203561782837 0.98425191640853882 |}];
  eval_and_print ~target:1.;
  [%expect {| 0.64496642351150513 0.64496642351150513 0.64496642351150513 |}]

let%expect_test _ =
  let vs = Tensor.of_float1 [| -1.01; -1.; -0.99; 0.5; 0.25; 0.; 2.; 4.; -1.; -3. |] in
  Stdio.printf
    !"%{sexp:float array} %{sexp:float}\n"
    Tensor.(huber_loss vs (Tensor.f 0.) ~reduction:None |> to_float1_exn)
    Tensor.(huber_loss vs (Tensor.f 0.) |> to_float0_exn);
  [%expect
    {| (0.50999999046325684 0.5 0.49005001783370972 0.125 0.03125 0 1.5 3.5 0.5 2.5) 0.9656299352645874 |}]

let%expect_test _ =
  let vs = List.range 1 10 |> Array.of_list |> Tensor.of_int1 in
  let chunk = Tensor.chunk vs ~chunks:4 ~dim:0 in
  Stdio.printf
    !"%{sexp:int array} %{sexp:int array list}\n"
    (Tensor.to_int1_exn vs)
    (List.map chunk ~f:Tensor.to_int1_exn);
  [%expect {| (1 2 3 4 5 6 7 8 9) ((1 2 3) (4 5 6) (7 8 9)) |}]

let%expect_test _ =
  let vs = Tensor.of_int1 [| 3; 1; 4 |] in
  let ws = Tensor.to_type vs ~type_:(T Float) in
  let xs = Tensor.reshape vs ~shape:[ -1; 1 ] in
  Stdio.printf
    "%b %b %b %b\n"
    (Tensor.eq vs vs)
    (Tensor.eq vs ws)
    (Tensor.eq ws ws)
    (Tensor.eq vs xs);
  [%expect {| true false true false |}];
  let ws = Tensor.of_int1 [| 3; 1 |] in
  let xs = Tensor.of_int1 [| 4; 2; 5 |] in
  Stdio.printf "%b %b\n" (Tensor.eq vs ws) (Tensor.eq vs xs);
  [%expect {| false false |}];
  Tensor.(xs -= of_int0 1);
  Stdio.printf "%b %b\n" (Tensor.eq vs ws) (Tensor.eq vs xs);
  [%expect {| false true |}]

let%expect_test _ =
  let t = Tensor.of_int2 [| [| 3; 1; 4 |]; [| 1; 5; 9 |] |] in
  Tensor.to_list t
  |> List.iter ~f:(fun t -> Tensor.to_int1_exn t |> Stdio.printf !"%{sexp:int array}\n");
  [%expect {|
    (3 1 4)
    (1 5 9) |}];
  assert (Tensor.device t = Cpu)

let%expect_test _ =
  (* Element-wise squaring of a vector. *)
  let t = Tensor.of_float1 [| 1.; 2.; 3. |] in
  let t = Tensor.einsum ~equation:"i, i -> i" [ t; t ] in
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn t);
  (* Matrix transpose *)
  let t = Tensor.of_int2 [| [| 3; 1; 4 |]; [| 1; 5; 9 |] |] in
  let t = Tensor.einsum ~equation:"ij -> ji" [ t ] in
  Tensor.to_list t
  |> List.iter ~f:(fun t -> Tensor.to_int1_exn t |> Stdio.printf !"%{sexp:int array}\n");
  (* Sum all elements *)
  let t = Tensor.einsum ~equation:"ij -> " [ t ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
  [%expect {|
    (1 4 9)
    (3 1)
    (1 5)
    (4 9)
    23 |}]
