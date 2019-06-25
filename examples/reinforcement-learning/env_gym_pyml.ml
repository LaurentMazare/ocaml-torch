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
  ; action_repeat : int
  }

let create str ~action_repeat =
  if not (Py.is_initialized ()) then Py.initialize ();
  let gym = Py.import "gym" in
  let env = Py.Module.get_function gym "make" [| Py.String.of_string str |] in
  let np = Py.import "numpy" in
  { env; np; action_repeat }

let to_tensor t np_array =
  let np_array = Py.Module.get_function t.np "ascontiguousarray" [| np_array |] in
  Py.Object.call_method np_array "astype" [| Py.Module.get t.np "float32" |]
  |> Numpy.to_bigarray Float32 C_layout
  |> Tensor.of_bigarray
  |> Tensor.to_type ~type_:(T Float)

let reset t =
  let reset_fn = Py.Object.get_attr_string t.env "reset" in
  Py.Callable.to_function (Option.value_exn reset_fn) [||] |> to_tensor t

let one_step t ~action =
  let v = Py.Object.call_method t.env "step" [| Py.Int.of_int action |] in
  let obs, reward, is_done, _ = Py.Tuple.to_tuple4 v in
  { obs = to_tensor t obs
  ; reward = Py.Float.to_float reward
  ; is_done = Py.Bool.to_bool is_done
  }

let step t ~action =
  let rec loop acc_rewards step_left =
    let { obs; reward; is_done } = one_step t ~action in
    let reward = acc_rewards +. reward in
    if is_done || step_left = 0
    then { obs; reward; is_done }
    else loop reward (step_left - 1)
  in
  loop 0. t.action_repeat

let actions t =
  Py.Object.call_function_obj_args
    Pyops.((t.env.@$("unwrapped")).@$("get_action_meanings"))
    [||]
  |> Py.Sequence.to_list_map Py.String.to_string

let lives t =
  Py.Object.call_function_obj_args
    Pyops.(((t.env.@$("unwrapped")).@$("ale")).@$("lives"))
    [||]
  |> Py.Int.to_int
