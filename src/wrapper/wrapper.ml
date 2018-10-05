open! Ctypes

module C = Torch_bindings.C(Torch_generated)

module Kind = struct
  type t =
    | Uint8
    | Int8
    | Int16
    | Int
    | Int64
    | Half
    | Float
    | Double
    | ComplexHalf
    | ComplexFloat
    | ComplexDouble

  (* Hardcoded, should match ScalarType.h *)
  let to_int = function
    | Uint8 -> 0
    | Int8 -> 1
    | Int16 -> 2
    | Int -> 3
    | Int64 -> 4
    | Half -> 5
    | Float -> 6
    | Double -> 7
    | ComplexHalf -> 8
    | ComplexFloat -> 9
    | ComplexDouble -> 10

  let of_int_exn = function
    | 0 -> Uint8
    | 1 -> Int8
    | 2 -> Int16
    | 3 -> Int
    | 4 -> Int64
    | 5 -> Half
    | 6 -> Float
    | 7 -> Double
    | 8 -> ComplexHalf
    | 9 -> ComplexFloat
    | 10 -> ComplexDouble
    | d -> failwith (Printf.sprintf "unexpected kind %d" d)
end

module Tensor = struct
  open! C.Tensor
  type nonrec t = t

  let float_vec ?(kind = `float) values =
    let values_len = List.length values in
    let values = CArray.of_list double values |> CArray.start in
    let kind =
      match kind with
      | `float -> Kind.Float
      | `double -> Double
      | `half -> Half
    in
    let t = float_vec values values_len (Kind.to_int kind) in
    Gc.finalise free t;
    t

  let int_vec ?(kind = `int) values =
    let values_len = List.length values in
    let values =
      List.map Int64.of_int values
      |> CArray.of_list int64_t
      |> CArray.start
    in
    let kind =
      match kind with
      | `uint8 -> Kind.Uint8
      | `int8 -> Int8
      | `int16 -> Int16
      | `int -> Int
      | `int64 -> Int64
    in
    let t = int_vec values values_len (Kind.to_int kind) in
    Gc.finalise free t;
    t

  let zeros ?(kind=Kind.Float) dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let t = zeros dim_array (List.length dims) (Kind.to_int kind) in
    Gc.finalise free t;
    t

  let ones ?(kind=Kind.Float) dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let t = ones dim_array (List.length dims) (Kind.to_int kind) in
    Gc.finalise free t;
    t

  let rand dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let t = rand dim_array (List.length dims) in
    Gc.finalise free t;
    t

  let reshape t ~dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let t = reshape t dim_array (List.length dims) in
    Gc.finalise free t;
    t

  let shape t =
    let num_dims = num_dims t in
    let carray = CArray.make int num_dims in
    shape t (CArray.start carray);
    CArray.to_list carray

  let kind t = scalar_type t |> Kind.of_int_exn

  let add x y =
    let t = add x y in
    Gc.finalise free t;
    t

  let sub x y =
    let t = sub x y in
    Gc.finalise free t;
    t

  let mul x y =
    let t = mul x y in
    Gc.finalise free t;
    t

  let div x y =
    let t = div x y in
    Gc.finalise free t;
    t

  let pow x y =
    let t = pow x y in
    Gc.finalise free t;
    t

  let matmul x y =
    let t = matmul x y in
    Gc.finalise free t;
    t

  let grad t =
    let grad = grad t in
    Gc.finalise free grad;
    grad

  let requires_grad t =
    if requires_grad t <> 0
    then true
    else false

  let set_requires_grad t ~b =
    let t = set_requires_grad t (if b then 1 else 0) in
    Gc.finalise free t;
    t

  let get t index =
    let t = get t index in
    Gc.finalise free t;
    t

  let select t ~dim ~index =
    let t = select t dim index in
    Gc.finalise free t;
    t

  let float_value t = double_value t
  let int_value t = int64_value t |> Int64.to_int
  let fill_float t v = fill_double t v
  let fill_int t i = fill_int64 t (Int64.of_int i)

  let set_float2 = set_double2

  let backward = backward
  let print = print
  let save t filename = save t filename
  let load filename =
    let t = load filename in
    Gc.finalise free t;
    t

  let sum t =
    let t = sum t in
    Gc.finalise free t;
    t

  let mean t =
    let t = mean t in
    Gc.finalise free t;
    t

  let argmax t =
    let t = argmax t in
    Gc.finalise free t;
    t

  let softmax t =
    let t = softmax t in
    Gc.finalise free t;
    t

  let neg t =
    let t = neg t in
    Gc.finalise free t;
    t

  let log t =
    let t = log t in
    Gc.finalise free t;
    t

  let eq t1 t2 =
    let t = eq t1 t2 in
    Gc.finalise free t;
    t

  let sub_assign = sub_assign
end
