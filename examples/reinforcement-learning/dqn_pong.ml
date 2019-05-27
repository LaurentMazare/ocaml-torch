open Base
open Torch

let total_episodes = 500
let update_target_every = 1_000
let verbose = false
let render = false

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

let model vs actions =
  let linear1 = Layer.linear vs ~input_dim:(80 * 80) 200 in
  let linear2 = Layer.linear vs ~input_dim:200 actions in
  Layer.of_fn (fun xs ->
      Tensor.flatten xs |> Layer.apply linear1 |> Tensor.relu |> Layer.apply linear2)

module DqnAgent : sig
  type t

  val create : actions:int -> memory_capacity:int -> t
  val action : t -> state -> total_frames:int -> int
  val experience_replay : t -> unit
  val transition_feedback : t -> Transition.t -> unit
  val update_target_model : t -> unit
  val var_store : t -> Var_store.t
end = struct
  type t =
    { model : Layer.t
    ; target_model : Layer.t
    ; vs : Var_store.t
    ; target_vs : Var_store.t
    ; memory : Transition.t Replay_memory.t
    ; actions : int
    ; batch_size : int
    ; gamma : float
    ; optimizer : Optimizer.t
    }

  let create ~actions ~memory_capacity =
    let target_vs = Var_store.create ~frozen:true ~name:"target-dqn" () in
    let target_model = model target_vs actions in
    let vs = Var_store.create ~name:"dqn" () in
    let model = model vs actions in
    let memory = Replay_memory.create ~capacity:memory_capacity in
    let optimizer =
      Optimizer.rmsprop vs ~learning_rate:0.0025 ~alpha:0.9 ~eps:0.01 ~momentum:0.0
    in
    { model
    ; target_model
    ; vs
    ; target_vs
    ; memory
    ; actions
    ; batch_size = 64
    ; gamma = 0.99
    ; optimizer
    }

  let var_store t = t.vs

  (* Only store the main network weights. *)

  let update_target_model t = Var_store.copy ~src:t.vs ~dst:t.target_vs

  let action t state ~total_frames =
    (* epsilon-greedy action choice. *)
    let epsilon = Float.max 0.02 (0.5 -. (Float.of_int total_frames /. 1_000_000.)) in
    if Float.( < ) epsilon (Random.float 1.)
    then (
      let qvalues =
        Tensor.no_grad (fun () -> Tensor.unsqueeze state ~dim:0 |> Layer.apply t.model)
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
        Layer.apply t.model states
        |> Tensor.gather
             ~dim:1
             ~index:(Tensor.unsqueeze actions ~dim:1)
             ~sparse_grad:false
        |> Tensor.squeeze1 ~dim:1
      in
      let next_qvalues =
        Tensor.no_grad (fun () ->
            Layer.apply t.target_model next_states
            |> Tensor.max2 ~dim:1 ~keepdim:false
            |> fst)
      in
      let expected_qvalues = Tensor.(rewards + (f t.gamma * next_qvalues * continue)) in
      let loss = Tensor.mse_loss qvalues expected_qvalues in
      Optimizer.backward_step t.optimizer ~loss)

  let transition_feedback t transition = Replay_memory.push t.memory transition
end

(* Initial shape is (210, 160, 3), convert to (1, 80, 80) and take the diff. *)
let preprocess () =
  let prev_img = ref None in
  fun state ->
    let d i ~factor = Tensor.(select state ~dim:2 ~index:i * f (factor /. 255.)) in
    let img =
      (* RGB to grey conversion. *)
      Tensor.(d 0 ~factor:0.299 + d 1 ~factor:0.587 + d 2 ~factor:0.114)
      |> Tensor.narrow ~dim:0 ~start:35 ~length:160
      |> Tensor.slice ~dim:0 ~start:0 ~end_:160 ~step:2
      |> Tensor.slice ~dim:1 ~start:0 ~end_:160 ~step:2
      |> Tensor.unsqueeze ~dim:0
    in
    let diff =
      match !prev_img with
      | None -> Tensor.zeros_like img
      | Some prev_img -> Tensor.(img - prev_img)
    in
    prev_img := Some img;
    diff

let maybe_load_weights agent =
  match Sys.argv with
  | [| _ |] -> ()
  | [| _; filename |] ->
    Serialize.load_multi_
      ~named_tensors:(DqnAgent.var_store agent |> Var_store.all_vars)
      ~filename;
    DqnAgent.update_target_model agent
  | _ -> Printf.failwithf "usage: %s [weights]" Sys.argv.(0) ()

let () =
  let module E = Env_gym_pyml in
  let env = E.create "PongDeterministic-v4" ~action_repeat:1 in
  let agent = DqnAgent.create ~actions:2 ~memory_capacity:50_000 in
  maybe_load_weights agent;
  let total_frames = ref 0 in
  for episode_idx = 1 to total_episodes do
    let episode_frames = ref 0 in
    let preprocess = preprocess () in
    let rec loop state acc_reward =
      let action = DqnAgent.action agent state ~total_frames:!total_frames in
      let { E.obs = next_state; reward; is_done } = E.step env ~action:(2 + action) in
      if render
      then
        Torch_vision.Image.write_image
          next_state
          ~filename:(Printf.sprintf "out%d.png" !total_frames);
      let next_state = preprocess next_state in
      DqnAgent.transition_feedback agent { state; action; next_state; reward; is_done };
      DqnAgent.experience_replay agent;
      Caml.Gc.full_major ();
      Int.incr total_frames;
      Int.incr episode_frames;
      if !total_frames % update_target_every = 0 then DqnAgent.update_target_model agent;
      let acc_reward = reward +. acc_reward in
      if verbose && Float.( <> ) reward 0.
      then
        Stdio.printf
          "reward: %4.0f total: %6.0f (%d frames)\n%!"
          reward
          acc_reward
          !total_frames;
      if is_done then acc_reward else loop next_state acc_reward
    in
    let reward = loop (E.reset env |> preprocess) 0. in
    Stdio.printf
      "%d %f (%d/%d frames)\n%!"
      episode_idx
      reward
      !episode_frames
      !total_frames;
    if episode_idx % 10 = 0
    then
      Serialize.save_multi
        ~named_tensors:(DqnAgent.var_store agent |> Var_store.all_vars)
        ~filename:(Printf.sprintf "dqn-pong-%d.ckpt" episode_idx)
  done
