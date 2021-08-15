(* This is adapted from OpenAI Spinning Up series:
   https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

   A TensorFlow Python implementation can be found here:
   https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py
*)

open Base
open Torch

let env_name = "CartPole-v0"
let obs_dim = 4
let n_acts = 2
let batch_size = 5000
let learning_rate = 1e-2
let epochs = 50

let model vs ~obs_dim ~n_acts =
  let linear1 = Layer.linear vs ~input_dim:obs_dim 32 in
  let linear2 = Layer.linear vs ~input_dim:32 n_acts in
  fun xs -> Layer.forward linear1 xs |> Tensor.tanh |> Layer.forward linear2

module Acc = struct
  type t =
    { mutable acc_obs : Tensor.t list
    ; mutable acc_acts : int list
    ; mutable episode_rewards : float list
    ; mutable acc_rewards : float list
    ; mutable sum_rewards : float
    ; mutable episodes : int
    }

  let create () =
    { acc_obs = []
    ; acc_acts = []
    ; episode_rewards = []
    ; acc_rewards = []
    ; sum_rewards = 0.
    ; episodes = 0
    }

  let update t ~obs ~action ~reward =
    t.acc_obs <- Tensor.copy obs :: t.acc_obs;
    t.acc_acts <- action :: t.acc_acts;
    t.episode_rewards <- reward :: t.episode_rewards;
    t.sum_rewards <- reward +. t.sum_rewards

  let is_done t =
    t.episodes <- t.episodes + 1;
    let _, episode_rewards =
      List.fold t.episode_rewards ~init:(0., []) ~f:(fun (acc_r, acc) reward ->
          let acc_r = acc_r +. reward in
          acc_r, acc_r :: acc)
    in
    t.acc_rewards <- List.rev_append episode_rewards t.acc_rewards;
    t.episode_rewards <- []

  let length t = List.length t.acc_obs
end

let () =
  let module E = Env_gym_pyml in
  let env = E.create env_name ~action_repeat:1 in
  let vs = Var_store.create ~name:"pg" () in
  let model = model vs ~obs_dim ~n_acts in
  let optimizer = Optimizer.adam vs ~learning_rate in
  for epoch_idx = 1 to epochs do
    let acc = Acc.create () in
    let rec loop obs =
      let action =
        Tensor.no_grad (fun () ->
            model (Tensor.unsqueeze obs ~dim:0)
            |> Tensor.softmax ~dim:1 ~dtype:(T Float)
            |> Tensor.multinomial ~num_samples:1 ~replacement:true
            |> Tensor.view ~size:[ 1 ]
            |> Tensor.to_int0_exn)
      in
      let { E.obs = next_obs; reward; is_done } = E.step env ~action in
      Acc.update acc ~obs ~action ~reward;
      if is_done
      then (
        let obs = E.reset env in
        Acc.is_done acc;
        if Acc.length acc <= batch_size then loop obs)
      else loop next_obs
    in
    loop (E.reset env);
    let batch_size = Acc.length acc in
    let acc_actions =
      Array.of_list acc.acc_acts |> Tensor.of_int1 |> Tensor.unsqueeze ~dim:1
    in
    let acc_weights = Array.of_list acc.acc_rewards |> Tensor.of_float1 in
    let action_mask =
      Tensor.zeros [ batch_size; n_acts ]
      |> Tensor.scatter_ ~dim:1 ~src:(Tensor.f 1.) ~index:acc_actions
    in
    let logits = model (Tensor.stack acc.acc_obs ~dim:0) in
    let log_probs =
      Tensor.(
        sum_dim_intlist
          (action_mask * log_softmax logits ~dim:1 ~dtype:(T Float))
          ~dim:[ 1 ]
          ~keepdim:false
          ~dtype:(T Float))
    in
    let loss = Tensor.(~-(mean (acc_weights * log_probs))) in
    Optimizer.backward_step optimizer ~loss;
    Stdio.printf
      "%d %d %f\n%!"
      epoch_idx
      acc.episodes
      (acc.sum_rewards /. Float.of_int acc.episodes)
  done
