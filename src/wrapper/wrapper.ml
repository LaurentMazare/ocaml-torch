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

  let requires_grad t =
    if requires_grad t <> 0
    then true
    else false

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
  let sum = sum2
  let mean = mean2

  let argmax t = argmax1 t (-1) false

  let softmax t = softmax t (-1)
end

module Scalar = struct
  include Wrapper_generated.C.Scalar

  let int i =
    let t = int (Int64.of_int i) in
    Gc.finalise free t;
    t

  let float f =
    let t = float f in
    Gc.finalise free t;
    t
end

module Optimizer = struct
  include Wrapper_generated.C.Optimizer

  let adam tensors ~learning_rate =
    let t =
      adam
        CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
        (List.length tensors)
        learning_rate
    in
    Gc.finalise free t;
    t
end

module Serialize = struct
  include Wrapper_generated.C.Serialize

  let save t ~filename = save t filename
  let load ~filename =
    let t = load filename in
    Gc.finalise Wrapper_generated.C.Tensor.free t;
    t

  let save_multi ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    save_multi
      CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
      CArray.(of_list string names |> start)
      (List.length named_tensors)
      filename

  let load_multi ~names ~filename =
    let ntensors = List.length names in
    let tensors = CArray.make Wrapper_generated.C.Tensor.t ntensors in
    load_multi
      (CArray.start tensors)
      CArray.(of_list string names |> start)
      ntensors
      filename;
    let tensors = CArray.to_list tensors in
    List.iter (Gc.finalise Wrapper_generated.C.Tensor.free) tensors;
    tensors

  let load_multi_ ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    load_multi_
      CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
      CArray.(of_list string names |> start)
      (List.length named_tensors)
      filename
end
