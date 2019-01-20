open Base
open Openai_gym
open Torch

let tensor_of_observation obs =
  Yojson.Basic.Util.to_list obs
  |> List.map ~f:Yojson.Basic.Util.to_number
  |> Tensor.float_vec

type t = Gym_t.instance_id

let create str =
  let t = Gym_client.env_create str in
  Caml.Gc.finalise Gym_client.env_close t;
  t

let reset t =
  let init = Gym_client.env_reset t in
  tensor_of_observation init.observation

let step t ~action ~render =
  let r = Gym_client.env_step t { action } render in
  { Env_intf.obs = tensor_of_observation r.step_observation
  ; reward = r.step_reward
  ; is_done = r.step_done
  }
