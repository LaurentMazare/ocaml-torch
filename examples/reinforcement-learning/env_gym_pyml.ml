open Base
open Torch

type t = Pytypes.pyobject

let create str =
  if not (Py.is_initialized ())
  then Py.initialize ();
  let gym = Py.import "gym" in
  Py.Module.get_function gym "make" [| Py.String.of_string str |]

let to_tensor np = Numpy.to_bigarray Float32 C_layout np |> Tensor.of_bigarray

let reset t =
  Py.Object.call_method t "reset" [||]
  |> to_tensor

let step t ~action ~render:_ =
  let v = Py.Object.call_method t "step" [| Py.Int.of_int action |] in
  let obs, reward, is_done, _ = Py.Tuple.to_tuple4 v in
  { Env_intf.obs = to_tensor obs
  ; reward = Py.Float.to_float reward
  ; is_done = Py.Bool.to_bool is_done
  }
