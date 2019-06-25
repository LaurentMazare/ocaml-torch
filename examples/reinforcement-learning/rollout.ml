open Base
open Torch
module E = Vec_env_gym_pyml

module Frame_stack : sig
  type t

  val create : num_procs:int -> num_stack:int -> t
  val update : t -> ?masks:Tensor.t -> Tensor.t -> Tensor.t
end = struct
  type t =
    { data : Tensor.t
    ; num_procs : int
    ; num_stack : int
    }

  let create ~num_procs ~num_stack =
    { data = Tensor.zeros [ num_procs; num_stack; 84; 84 ] ~kind:(T Float)
    ; num_procs
    ; num_stack
    }

  let update t ?masks img =
    Option.iter masks ~f:(fun masks ->
        Tensor.(t.data *= view masks ~size:[ t.num_procs; 1; 1; 1 ]));
    let stack_slice i = Tensor.narrow t.data ~dim:1 ~start:i ~length:1 in
    for frame_index = 1 to t.num_stack - 1 do
      Tensor.copy_ (stack_slice (frame_index - 1)) ~src:(stack_slice frame_index)
    done;
    Tensor.copy_ (stack_slice (t.num_stack - 1)) ~src:img;
    t.data
end

type actor_critic =
  { actor : Tensor.t
  ; critic : Tensor.t
  }

type totals =
  { rewards : float
  ; episodes : float
  }

type rollout =
  { states : Tensor.t
  ; returns : Tensor.t
  ; actions : Tensor.t
  ; values : Tensor.t
  }

type t =
  { envs : E.t
  ; num_steps : int
  ; num_procs : int
  ; frame_stack : Frame_stack.t
  ; s_states : Tensor.t
  ; sum_rewards : Tensor.t
  ; mutable total_rewards : float
  ; mutable total_episodes : float
  }

let create ~atari_game ~num_steps ~num_stack ~num_procs =
  let frame_stack = Frame_stack.create ~num_procs ~num_stack in
  let envs = E.create atari_game ~num_processes:num_procs in
  let obs = E.reset envs in
  Tensor.print_shape obs ~name:"obs";
  ignore (Frame_stack.update frame_stack obs : Tensor.t);
  let s_states =
    Tensor.zeros [ num_steps + 1; num_procs; num_stack; 84; 84 ] ~kind:(T Float)
  in
  { envs
  ; num_steps
  ; num_procs
  ; frame_stack
  ; s_states
  ; sum_rewards = Tensor.zeros [ num_procs ]
  ; total_rewards = 0.
  ; total_episodes = 0.
  }

let set tensor i src = Tensor.copy_ (Tensor.get tensor i) ~src
let action_space t = E.action_space t.envs

let run t ~model =
  set t.s_states 0 (Tensor.get t.s_states (-1));
  let s_values = Tensor.zeros [ t.num_steps; t.num_procs ] in
  let s_rewards = Tensor.zeros [ t.num_steps; t.num_procs ] in
  let s_actions = Tensor.zeros [ t.num_steps; t.num_procs ] ~kind:(T Int64) in
  let s_masks = Tensor.zeros [ t.num_steps; t.num_procs ] in
  for s = 0 to t.num_steps - 1 do
    let { actor; critic } = Tensor.no_grad (fun () -> model (Tensor.get t.s_states s)) in
    let probs = Tensor.softmax actor ~dim:(-1) in
    let actions =
      Tensor.multinomial probs ~num_samples:1 ~replacement:true |> Tensor.squeeze_last
    in
    let { E.obs; reward; is_done } =
      E.step t.envs ~actions:(Tensor.to_int1_exn actions |> Array.to_list)
    in
    Tensor.(t.sum_rewards += reward);
    t.total_rewards
    <- (t.total_rewards +. Tensor.(sum (t.sum_rewards * is_done) |> to_float0_exn));
    t.total_episodes <- (t.total_episodes +. Tensor.(sum is_done |> to_float0_exn));
    let masks = Tensor.(f 1. - is_done) in
    Tensor.(t.sum_rewards *= masks);
    let obs = Frame_stack.update t.frame_stack obs ~masks in
    set s_actions s actions;
    set s_values s (critic |> Tensor.squeeze1 ~dim:(-1));
    set t.s_states (s + 1) obs;
    set s_rewards s reward;
    set s_masks s masks
  done;
  let s_returns =
    let r = Tensor.zeros [ t.num_steps + 1; t.num_procs ] in
    let critic =
      Tensor.no_grad (fun () -> (model (Tensor.get t.s_states (-1))).critic)
    in
    set r (-1) (Tensor.view critic ~size:[ t.num_procs ]);
    for s = t.num_steps - 1 downto 0 do
      set r s Tensor.((get r Int.(s + 1) * f 0.99 * get s_masks s) + get s_rewards s)
    done;
    r
  in
  { states = t.s_states; returns = s_returns; actions = s_actions; values = s_values }

let get_and_reset_totals t =
  let res = { rewards = t.total_rewards; episodes = t.total_episodes } in
  t.total_rewards <- 0.;
  t.total_episodes <- 0.;
  res
