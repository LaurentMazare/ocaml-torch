open Base
open Torch

type step =
  { obs : Tensor.t
  ; reward : Tensor.t
  ; is_done : Tensor.t
  }

type t =
  { envs : Pytypes.pyobject
  ; np : Pytypes.pyobject
  }

let create str ~num_processes =
  if not (Py.is_initialized ())
  then (
    Py.add_python_path "examples/reinforcement-learning";
    Py.initialize ());
  let wrappers = Py.import "atari_wrappers" in
  let envs =
    Py.Module.get_function
      wrappers
      "make"
      [| Py.String.of_string str; Py.Int.of_int num_processes |]
  in
  let np = Py.import "numpy" in
  { envs; np }

let to_tensor t np_array =
  let np_array = Py.Module.get_function t.np "ascontiguousarray" [| np_array |] in
  Py.Object.call_method np_array "astype" [| Py.Module.get t.np "float32" |]
  |> Numpy.to_bigarray Float32 C_layout
  |> Tensor.of_bigarray
  |> Tensor.to_type ~type_:Float

let reset t =
  let reset_fn = Py.Object.get_attr_string t.envs "reset" in
  Py.Callable.to_function (Option.value_exn reset_fn) [||] |> to_tensor t

let step t ~actions =
  let v =
    Py.Object.call_method t.envs "step" [| Py.List.of_list_map Py.Int.of_int actions |]
  in
  let obs, reward, is_done, _ = Py.Tuple.to_tuple4 v in
  { obs = to_tensor t obs; reward = to_tensor t reward; is_done = to_tensor t is_done }

let action_space t =
  let action_space =
    Option.value_exn (Py.Object.get_attr_string t.envs "action_space")
  in
  Option.value_exn (Py.Object.get_attr_string action_space "n") |> Py.Int.to_int
