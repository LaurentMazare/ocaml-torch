open Ctypes

module Tensor = struct
  include Wrapper_generated
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

  let ones ?(kind = Kind.Float) dims = ones dims kind
  let zeros ?(kind = Kind.Float) dims = zeros dims kind
  let rand ?(kind = Kind.Float) dims = rand dims kind

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

  let reshape t ~dims = reshape t dims

  let shape t =
    let num_dims = num_dims t in
    let carray = CArray.make int num_dims in
    shape t (CArray.start carray);
    CArray.to_list carray

  let kind t = scalar_type t |> Kind.of_int_exn

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

  let sum = sum1
  let mean = mean1

  let argmax t = argmax1 t (Int64.of_int (-1)) false

  let softmax t = softmax t (Int64.of_int (-1))

  let neg t =
    let t = neg t in
    Gc.finalise free t;
    t

  let eq t1 t2 =
    let t = eq t1 t2 in
    Gc.finalise free t;
    t

  let sub_assign = sub_assign
end
