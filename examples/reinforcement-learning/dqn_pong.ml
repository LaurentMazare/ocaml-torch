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

  let batch_states ts =
    List.map ts ~f:(fun t -> t.state)
    |> Tensor.stack ~dim:0

  let batch_next_states ts =
    List.map ts ~f:(fun t -> t.next_state)
    |> Tensor.stack ~dim:0

  let batch_rewards ts =
    List.map ts ~f:(fun t -> t.reward)
    |> Array.of_list
    |> Tensor.of_float1

  let batch_actions ts =
    List.map ts ~f:(fun t -> t.action)
    |> Array.of_list
    |> Tensor.of_int1

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

  let create ~capacity =
    { memory = Queue.create ()
    ; capacity
    ; position = 0
    }

  let length t = Queue.length t.memory

  let push t elem =
    if Queue.length t.memory < t.capacity
    then begin
      Queue.enqueue t.memory elem;
    end else begin
      Queue.set t.memory t.position elem
    end;
    t.position <- (t.position + 1) % t.capacity

  let sample t ~batch_size =
    List.init batch_size ~f:(fun _ ->
      let index = Random.int (Queue.length t.memory) in
      Queue.get t.memory index)
end

let cnn_model vs actions =
  let conv2d = Layer.conv2d_ vs ~ksize:5 ~stride:2 in
  let conv1 = conv2d ~input_dim:1 16 in
  let bn1 = Layer.batch_norm2d vs 16 in
  let conv2 = conv2d ~input_dim:16 32 in
  let bn2 = Layer.batch_norm2d vs 32 in
  let conv3 = conv2d ~input_dim:32 32 in
  let bn3 = Layer.batch_norm2d vs 32 in
  let linear = Layer.linear vs ~input_dim:(7 * 7 * 32) actions in
  Layer.of_fn_ (fun xs ~is_training ->
    Layer.apply conv1 xs
    |> Layer.apply_ bn1 ~is_training
    |> Tensor.relu
    |> Layer.apply conv2
    |> Layer.apply_ bn2 ~is_training
    |> Tensor.relu
    |> Layer.apply conv3
    |> Layer.apply_ bn3 ~is_training
    |> Tensor.relu
    |> Tensor.flatten
    |> Layer.apply linear)

module DqnAgent : sig
  type t
  val create : actions:int -> memory_capacity:int -> t
  val action : t -> state -> int
  val experience_replay : t -> unit
  val transition_feedback : t -> Transition.t -> unit
end = struct
  type t =
    { model : Layer.t_with_training
    ; memory : Transition.t Replay_memory.t
    ; actions : int
    ; batch_size : int
    ; gamma : float
    ; epsilon_decay : float
    ; epsilon_min : float
    ; mutable epsilon : float
    ; optimizer : Optimizer.t
    }

  let create ~actions ~memory_capacity =
    let vs = Var_store.create ~name:"dqn" () in
    let model = cnn_model vs actions in
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
    if Float.(<) t.epsilon (Random.float 1.)
    then begin
      let qvalues =
        Tensor.no_grad (fun () ->
          Tensor.unsqueeze state ~dim:0
          |> Layer.apply_ t.model ~is_training:false)
      in
      Tensor.argmax1 qvalues ~dim:1 ~keepdim:false
      |> Tensor.to_int1_exn
      |> fun xs -> xs.(0)
    end else Random.int t.actions

  let experience_replay t =
    if t.batch_size <= Replay_memory.length t.memory
    then begin
      let transitions = Replay_memory.sample t.memory ~batch_size:t.batch_size in
      let states = Transition.batch_states transitions in
      let next_states = Transition.batch_next_states transitions in
      let actions = Transition.batch_actions transitions in
      let rewards = Transition.batch_rewards transitions in
      let continue = Transition.batch_continue transitions in
      let qvalues =
        Layer.apply_ t.model states ~is_training:true
        |> Tensor.gather ~dim:1 ~index:(Tensor.unsqueeze actions ~dim:1)
        |> Tensor.squeeze1 ~dim:1
      in
      let next_qvalues =
        Tensor.no_grad (fun () ->
          Layer.apply_ t.model next_states ~is_training:false
          |> Tensor.max2 ~dim:1 ~keepdim:false
          |> fst)
      in
      let expected_qvalues = Tensor.(rewards + f t.gamma * next_qvalues * continue) in
      let loss = Tensor.mse_loss qvalues expected_qvalues in
      Optimizer.backward_step t.optimizer ~loss;
      t.epsilon <- Float.max t.epsilon_min (t.epsilon *. t.epsilon_decay)
    end

  let transition_feedback t transition = Replay_memory.push t.memory transition
end

(* Initial shape is (210, 160, 3), convert to (1, 80, 80) and take the diff. *)
let preprocess () =
  let prev_img = ref None in
  fun state ->
    let d i ~factor = Tensor.(select state ~dim:2 ~index:i * f (factor /. 255.)) in
    let img =
      Tensor.(d 0 ~factor:0.299 + d 1 ~factor:0.587 + d 2 ~factor:0.114)
      |> Tensor.narrow ~dim:0 ~start:35 ~length:160
      |> Tensor.unsqueeze ~dim:0
      |> Tensor.avg_pool2d ~ksize:(2, 2)
    in
    let diff =
      match !prev_img with
      | None -> Tensor.zeros_like img
      | Some prev_img -> Tensor.(img - prev_img)
    in
    prev_img := Some img;
    diff

let () =
  let module E = Env_gym_pyml in
  let env = E.create "PongNoFrameskip-v3" in
  let agent = DqnAgent.create ~actions:2 ~memory_capacity:50_000 in
  for episode_idx = 1 to total_episodes do
    let preprocess = preprocess () in
    let rec loop state acc_reward =
      let action = DqnAgent.action agent state in
      let { Env_intf.obs = next_state; reward; is_done } = E.step env ~action ~render:false in
      let next_state = preprocess next_state in
      DqnAgent.transition_feedback agent { state; action; next_state; reward; is_done };
      DqnAgent.experience_replay agent;
      Caml.Gc.full_major ();
      let acc_reward = reward +. acc_reward in
      if is_done then acc_reward else loop next_state acc_reward
    in
    let reward = loop (E.reset env |> preprocess) 0. in
    Stdio.printf "%d %f\n%!" episode_idx reward;
  done
