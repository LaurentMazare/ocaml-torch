open Base
open Torch

let total_episodes = 500

type state = Tensor.t

module Transition = struct
  type t =
    { state : state
    ; action : int
    ; next_state : state
    ; reward : float
    ; is_done : bool
    }

  let batch_states ts = List.map ts ~f:(fun t -> t.state) |> Tensor.stack ~dim:0

  let batch_next_states ts =
    List.map ts ~f:(fun t -> t.next_state) |> Tensor.stack ~dim:0

  let batch_rewards ts =
    List.map ts ~f:(fun t -> t.reward) |> Array.of_list |> Tensor.of_float1

  let batch_actions ts =
    List.map ts ~f:(fun t -> t.action) |> Array.of_list |> Tensor.of_int1

  let batch_continue ts =
    List.map ts ~f:(fun t -> if t.is_done then 0. else 1.)
    |> Array.of_list
    |> Tensor.of_float1
end

module Replay_memory : sig
  type _ t

  val create : capacity:int -> _ t
  val push : 'a t -> 'a -> unit
  val sample : 'a t -> batch_size:int -> 'a list
  val length : _ t -> int
end = struct
  type 'a t =
    { memory : 'a Queue.t
    ; capacity : int
    ; mutable position : int
    }

  let create ~capacity = { memory = Queue.create (); capacity; position = 0 }
  let length t = Queue.length t.memory

  let push t elem =
    if Queue.length t.memory < t.capacity
    then Queue.enqueue t.memory elem
    else Queue.set t.memory t.position elem;
    t.position <- (t.position + 1) % t.capacity

  let sample t ~batch_size =
    List.init batch_size ~f:(fun _ ->
        let index = Random.int (Queue.length t.memory) in
        Queue.get t.memory index)
end

let linear_model vs ~input_dim actions_dim =
  let linear1 = Layer.linear vs ~input_dim 24 in
  let linear2 = Layer.linear vs ~input_dim:24 24 in
  let linear3 = Layer.linear vs ~input_dim:24 actions_dim in
  Layer.of_fn (fun xs ->
      Layer.forward linear1 xs
      |> Tensor.relu
      |> Layer.forward linear2
      |> Tensor.relu
      |> Layer.forward linear3)

module DqnAgent : sig
  type t

  val create : state_dim:int -> actions:int -> memory_capacity:int -> t
  val action : t -> state -> int
  val experience_replay : t -> unit
  val transition_feedback : t -> Transition.t -> unit
end = struct
  type t =
    { model : Layer.t
    ; memory : Transition.t Replay_memory.t
    ; actions : int
    ; batch_size : int
    ; gamma : float
    ; epsilon_decay : float
    ; epsilon_min : float
    ; mutable epsilon : float
    ; optimizer : Optimizer.t
    }

  let create ~state_dim ~actions ~memory_capacity =
    let vs = Var_store.create ~name:"dqn" () in
    let model = linear_model vs ~input_dim:state_dim actions in
    let memory = Replay_memory.create ~capacity:memory_capacity in
    let optimizer = Optimizer.adam vs ~learning_rate:1e-3 in
    { model
    ; memory
    ; actions
    ; batch_size = 32
    ; gamma = 0.99
    ; epsilon_decay = 0.995
    ; epsilon_min = 0.01
    ; epsilon = 1.
    ; optimizer
    }

  let action t state =
    (* epsilon-greedy action choice. *)
    if Float.( < ) t.epsilon (Random.float 1.)
    then (
      let qvalues =
        Tensor.no_grad (fun () -> Tensor.unsqueeze state ~dim:0 |> Layer.forward t.model)
      in
      Tensor.argmax qvalues ~dim:1 ~keepdim:false
      |> Tensor.to_int1_exn
      |> fun xs -> xs.(0))
    else Random.int t.actions

  let experience_replay t =
    if t.batch_size <= Replay_memory.length t.memory
    then (
      let transitions = Replay_memory.sample t.memory ~batch_size:t.batch_size in
      let states = Transition.batch_states transitions in
      let next_states = Transition.batch_next_states transitions in
      let actions = Transition.batch_actions transitions in
      let rewards = Transition.batch_rewards transitions in
      let continue = Transition.batch_continue transitions in
      let qvalues =
        Layer.forward t.model states
        |> Tensor.gather
             ~dim:1
             ~index:(Tensor.unsqueeze actions ~dim:1)
             ~sparse_grad:false
        |> Tensor.squeeze1 ~dim:1
      in
      let next_qvalues =
        Tensor.no_grad (fun () ->
            Layer.forward t.model next_states |> Tensor.max2 ~dim:1 ~keepdim:false |> fst)
      in
      let expected_qvalues = Tensor.(rewards + (f t.gamma * next_qvalues * continue)) in
      let loss = Tensor.mse_loss qvalues expected_qvalues in
      Optimizer.backward_step t.optimizer ~loss;
      t.epsilon <- Float.max t.epsilon_min (t.epsilon *. t.epsilon_decay))

  let transition_feedback t transition = Replay_memory.push t.memory transition
end

(* Hard-code dimensions to CartPole-v1 for the time being. *)
let () =
  let module E = Env_gym_pyml in
  let env = E.create "CartPole-v1" ~action_repeat:1 in
  let agent = DqnAgent.create ~state_dim:4 ~actions:2 ~memory_capacity:1_000_000 in
  for episode_idx = 1 to total_episodes do
    let rec loop state acc_reward =
      let action = DqnAgent.action agent state in
      let { E.obs = next_state; reward; is_done } = E.step env ~action in
      DqnAgent.transition_feedback agent { state; action; next_state; reward; is_done };
      DqnAgent.experience_replay agent;
      let acc_reward = reward +. acc_reward in
      if is_done then acc_reward else loop next_state acc_reward
    in
    let reward = loop (E.reset env) 0. in
    Stdio.printf "%d %f\n%!" episode_idx reward
  done
