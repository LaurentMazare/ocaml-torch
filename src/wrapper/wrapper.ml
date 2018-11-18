open Ctypes

let ptr_of_string str =
  let len = String.length str in
  let carray = CArray.make Ctypes.char (1 + len) in
  String.iteri (fun i char -> CArray.set carray i char) str;
  CArray.set carray len '\x00';
  CArray.start carray

let ptr_of_strings strings =
  let strings = List.map ptr_of_string strings in
  let start = CArray.(of_list (ptr char) strings |> start) in
  Gc.finalise (fun _ -> ignore (Sys.opaque_identity strings : _ list)) start;
  start

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

  let of_bigarray (type a) (type b) (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let dims = Bigarray.Genarray.dims ga in
    let kind = Bigarray.Genarray.kind ga in
    let tensor_kind =
      match kind with
      | Bigarray.Float32 -> Kind.Float
      | Bigarray.Float64 -> Double
      | Bigarray.Int8_signed -> Int8
      | Bigarray.Int8_unsigned -> Uint8
      | Bigarray.Int16_signed -> Int16
      | Bigarray.Int32 -> Int
      | Bigarray.Int64 -> Int64
      | _ -> failwith "unsupported bigarray kind"
    in
    let t =
      tensor_of_data
        (bigarray_start genarray ga |> to_voidp)
        (Array.to_list dims |> List.map Signed.Long.of_int |> CArray.of_list long |> CArray.start)
        (Array.length dims)
        (Bigarray.kind_size_in_bytes kind)
        (Kind.to_int tensor_kind)
    in
    Gc.finalise free t;
    t

  let copy_to_bigarray (type a) (type b) t (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let kind = Bigarray.Genarray.kind ga in
    copy_data t
      (bigarray_start genarray ga |> to_voidp)
      (Bigarray.Genarray.dims ga |> Array.fold_left ( * ) 1 |> Int64.of_int)
      (Bigarray.kind_size_in_bytes kind)

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

  let float_value t = double_value t
  let int_value t = int64_value t |> Int64.to_int
  let fill_float t v = fill_double t v
  let fill_int t i = fill_int64 t (Int64.of_int i)

  let backward = backward
  let print = print
  let to_string t ~line_size = to_string t line_size
  let sum = sum2
  let mean = mean2
  let softmax = softmax2
  let log_softmax = log_softmax2
  let scatter_ = scatter_1

  let argmax t = argmax1 t ~dim:(-1) ~keepdim:false

  let copy_ t ~src = copy_ t src

  let defined = defined

  let new_tensor () =
    let t = new_tensor () in
    Gc.finalise free t;
    t
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

  let adam ~learning_rate =
    let t = adam learning_rate in
    Gc.finalise free t;
    t

  let sgd ~learning_rate ~momentum ~dampening ~weight_decay ~nesterov =
    let t =
      sgd
        learning_rate
        momentum
        dampening
        weight_decay
        nesterov
    in
    Gc.finalise free t;
    t

  let add_parameters t tensors =
    add_parameters
      t
      CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
      (List.length tensors)
end

module Serialize = struct
  include Wrapper_generated.C.Serialize

  let save t ~filename = save t filename

  let escape s = String.map (function | '.' -> '|' | c -> c) s
  let unescape s = String.map (function | '|' -> '.' | c -> c) s

  let load ~filename =
    let t = load filename in
    Gc.finalise Wrapper_generated.C.Tensor.free t;
    t

  let save_multi ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    let names = List.map escape names in
    save_multi
      CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
      (ptr_of_strings names)
      (List.length named_tensors)
      filename

  let load_multi ~names ~filename =
    let names = List.map escape names in
    let ntensors = List.length names in
    let tensors = CArray.make Wrapper_generated.C.Tensor.t ntensors in
    load_multi
      (CArray.start tensors)
      (ptr_of_strings names)
      ntensors
      filename;
    let tensors = CArray.to_list tensors in
    List.iter (Gc.finalise Wrapper_generated.C.Tensor.free) tensors;
    tensors

  let load_multi_ ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    let names = List.map escape names in
    load_multi_
      CArray.(of_list Wrapper_generated.C.Tensor.t tensors |> start)
      (ptr_of_strings names)
      (List.length named_tensors)
      filename

  let load_all ~filename =
    let all_tensors = ref [] in
    let callback =
      coerce
        (Foreign.funptr (string @-> Wrapper_generated.C.Tensor.t @-> returning void))
        (static_funptr (string @-> Wrapper_generated.C.Tensor.t @-> returning void))
        (fun tensor_name tensor ->
          Gc.finalise Wrapper_generated.C.Tensor.free tensor;
          all_tensors := (unescape tensor_name, tensor) :: !all_tensors)
    in
    load_callback filename callback;
    !all_tensors
end

module Cuda = struct
  include Wrapper_generated.C.Cuda
  let is_available () = is_available () <> 0
  let cudnn_is_available () = cudnn_is_available () <> 0
  let set_benchmark_cudnn b = set_benchmark_cudnn (if b then 1 else 0)
end

let manual_seed seed = Wrapper_generated.C.manual_seed (Int64.of_int seed)
