open Base
open Torch

type step =
  { obs : Tensor.t
  ; reward : float
  ; is_done : bool
  }

type t =
  { env : Pytypes.pyobject
  ; np : Pytypes.pyobject
  }

let create str =
  if not (Py.is_initialized ())
  then Py.initialize ();
  let gym = Py.import "gym" in
  let env = Py.Module.get_function gym "make" [| Py.String.of_string str |] in
  let np = Py.import "numpy" in
  { env; np }

let to_tensor t np_array =
  let np_array = Py.Module.get_function t.np "ascontiguousarray" [| np_array |] in
  Py.Object.call_method np_array "astype" [| Py.Module.get t.np "float32" |]
  |> Numpy.to_bigarray Float32 C_layout
  |> Tensor.of_bigarray
  |> Tensor.to_type ~type_:Float

let reset t =
  let reset_fn = Py.Object.get_attr_string t.env "reset" in
  Py.Callable.to_function (Option.value_exn reset_fn) [||]
  |> to_tensor t

let step t ~action =
  let v = Py.Object.call_method t.env "step" [| Py.Int.of_int action |] in
  let obs, reward, is_done, _ = Py.Tuple.to_tuple4 v in
  { obs = to_tensor t obs
  ; reward = Py.Float.to_float reward
  ; is_done = Py.Bool.to_bool is_done
  }
