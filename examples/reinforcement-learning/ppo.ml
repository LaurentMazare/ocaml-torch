(* Proximal Policy Optimization.
   https://arxiv.org/abs/1707.06347

   This algorithm is close to A2C and involves both an actor and
   a critic model that share some layers. Multiple rollouts are
   performed in parallel in the [Rollout] module.
   Then the actor and the critic are trained on this data. The
   main differences with A2C are that:
     - The action loss penalises getting far from the policy
       values used during rollout.
     - The advantage uses the Generalized Advantage Estimation.
       https://arxiv.org/abs/1506.02438
     - Rollouts are longer and return more data so the training
       part uses mini-batches from this data rather than all the
       data at once.

   https://spinningup.openai.com/en/latest/algorithms/ppo.html
*)
open Base
open Torch
module E = Vec_env_gym_pyml

let atari_game = "SpaceInvadersNoFrameskip-v4"

(* timesteps per actor/batch *)
let num_steps = 256

(* number of actors running in parallel *)
let num_procs = 8
let updates = 1_000_000
let num_stack = 4
let optim_batchsize = 64
let optim_epochs = 4

let model vs ~actions =
  let conv1 = Layer.conv2d_ vs ~ksize:8 ~stride:4 ~input_dim:num_stack 32 in
  let conv2 = Layer.conv2d_ vs ~ksize:4 ~stride:2 ~input_dim:32 64 in
  let conv3 = Layer.conv2d_ vs ~ksize:3 ~stride:1 ~input_dim:64 64 in
  let linear1 = Layer.linear vs ~input_dim:(64 * 7 * 7) 512 in
  let critic_linear = Layer.linear vs ~input_dim:512 1 in
  let actor_linear = Layer.linear vs ~input_dim:512 actions in
  fun xs ->
    let ys =
      Tensor.to_device xs ~device:(Var_store.device vs)
      |> Layer.forward conv1
      |> Tensor.relu
      |> Layer.forward conv2
      |> Tensor.relu
      |> Layer.forward conv3
      |> Tensor.relu
      |> Tensor.flatten
      |> Layer.forward linear1
      |> Tensor.relu
    in
    { Rollout.critic = Layer.forward critic_linear ys
    ; actor = Layer.forward actor_linear ys
    }

let train ~device =
  let rollout = Rollout.create ~atari_game ~num_steps ~num_stack ~num_procs in
  let action_space = Rollout.action_space rollout in
  Stdio.printf "Action space: %d\n%!" action_space;
  let vs = Var_store.create ~name:"a2c" () ~device in
  let model = model vs ~actions:action_space in
  let optimizer = Optimizer.adam vs ~learning_rate:1e-4 in
  let train_size = num_steps * num_procs in
  for index = 1 to updates do
    let { Rollout.states; actions; returns; values } = Rollout.run rollout ~model in
    let states =
      Tensor.narrow states ~dim:0 ~start:0 ~length:num_steps
      |> Tensor.view ~size:[ train_size; num_stack; 84; 84 ]
    in
    let actions = Tensor.view actions ~size:[ train_size ] in
    let _values = Tensor.view values ~size:[ train_size ] in
    let returns =
      Tensor.narrow returns ~dim:0 ~start:0 ~length:num_steps
      |> Tensor.view ~size:[ train_size; 1 ]
    in
    for _index = 1 to optim_epochs do
      let index =
        Tensor.randint ~high:train_size ~size:[ optim_batchsize ] ~options:(T Int64, Cpu)
      in
      let states = Tensor.index_select states ~dim:0 ~index in
      let actions = Tensor.index_select actions ~dim:0 ~index in
      let returns = Tensor.index_select returns ~dim:0 ~index in
      let { Rollout.actor; critic } = model states in
      let log_probs = Tensor.log_softmax actor ~dim:(-1) ~dtype:(T Float) in
      let probs = Tensor.softmax actor ~dim:(-1) ~dtype:(T Float) in
      let action_log_probs =
        let index = Tensor.unsqueeze actions ~dim:(-1) |> Tensor.to_device ~device in
        Tensor.gather log_probs ~dim:(-1) ~index ~sparse_grad:false |> Tensor.squeeze_last
      in
      let dist_entropy =
        Tensor.(
          ~-(log_probs * probs)
          |> sum1 ~dim:[ -1 ] ~keepdim:false ~dtype:(T Float)
          |> mean)
      in
      let advantages = Tensor.(to_device returns ~device - critic) in
      let value_loss = Tensor.(advantages * advantages) |> Tensor.mean in
      let action_loss = Tensor.(~-(detach advantages * action_log_probs) |> mean) in
      let loss = Tensor.(scale value_loss 0.5 + action_loss - scale dist_entropy 0.01) in
      Optimizer.backward_step optimizer ~loss ~clip_grad:(Value 0.5)
    done;
    Caml.Gc.full_major ();
    if index % 1_000 = 0
    then
      Serialize.save_multi
        ~named_tensors:(Var_store.all_vars vs)
        ~filename:(Printf.sprintf "a2c-%d.ckpt" index);
    if index % 25 = 0
    then (
      let { Rollout.rewards = r; episodes = e } = Rollout.get_and_reset_totals rollout in
      Stdio.printf "%d %f (%.0f episodes)\n%!" index (r /. e) e)
  done

let valid ~filename ~device =
  let rollout = Rollout.create ~atari_game ~num_steps:1000 ~num_stack ~num_procs:1 in
  let action_space = Rollout.action_space rollout in
  let vs = Var_store.create ~frozen:true ~name:"a2c" () ~device in
  let model = model vs ~actions:action_space in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename;
  let _ = Rollout.run rollout ~model in
  let { Rollout.rewards = r; episodes = e } = Rollout.get_and_reset_totals rollout in
  Stdio.printf "%f (%.0f episodes)\n%!" (r /. e) e

let () =
  Torch_core.Wrapper.manual_seed 42;
  let device = Device.cuda_if_available () in
  if Array.length Caml.Sys.argv > 1
  then valid ~filename:Caml.Sys.argv.(1) ~device
  else train ~device
