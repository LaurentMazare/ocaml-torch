open Base
open Torch

type t =
  { env : Pytypes.pyobject
  ; np_ascontiguousarray : Pytypes.pyobject array -> Pytypes.pyobject
  }

let create str =
  if not (Py.is_initialized ())
  then Py.initialize ();
  let gym = Py.import "gym" in
  let env = Py.Module.get_function gym "make" [| Py.String.of_string str |] in
  let np = Py.import "numpy" in
  let np_ascontiguousarray = Py.Module.get_function np "ascontiguousarray" in
  { env; np_ascontiguousarray }

let to_tensor ~np_ascontiguousarray np_array =
  np_ascontiguousarray [| np_array |]
  |> Numpy.to_bigarray Float64 C_layout
  |> Tensor.of_bigarray
  |> Tensor.to_type ~type_:Float

let reset t =
  let reset_fn = Py.Object.get_attr_string t.env "reset" in
  Py.Callable.to_function (Option.value_exn reset_fn) [||]
  |> to_tensor ~np_ascontiguousarray:t.np_ascontiguousarray

let step t ~action ~render:_ =
  let v = Py.Object.call_method t.env "step" [| Py.Int.of_int action |] in
  let obs, reward, is_done, _ = Py.Tuple.to_tuple4 v in
  { Env_intf.obs = to_tensor obs ~np_ascontiguousarray:t.np_ascontiguousarray
  ; reward = Py.Float.to_float reward
  ; is_done = Py.Bool.to_bool is_done
  }
