open Base
open Torch

let num_steps = 5
let updates = 1_000
let num_procs = 16
let num_stack = 4

type actor_critic =
  { actor : Tensor.t
  ; critic : Tensor.t }

let model vs ~actions =
  let conv1 = Layer.conv2d_ vs ~ksize:8 ~stride:4 ~input_dim:num_stack 32 in
  let conv2 = Layer.conv2d_ vs ~ksize:4 ~stride:2 ~input_dim:32 64 in
  let conv3 = Layer.conv2d_ vs ~ksize:3 ~stride:1 ~input_dim:64 64 in
  let linear1 = Layer.linear vs ~input_dim:(64 * 7 * 7) 512 in
  let critic_linear = Layer.linear vs ~input_dim:512 1 in
  let actor_linear = Layer.linear vs ~input_dim:512 actions in
  fun xs ->
    let ys =
      Tensor.(to_type xs ~type_:Float * f (1. /. 255.))
      |> Layer.apply conv1
      |> Tensor.relu
      |> Layer.apply conv2
      |> Tensor.relu
      |> Layer.apply conv3
      |> Tensor.relu
      |> Tensor.flatten
      |> Layer.apply linear1
      |> Tensor.relu
    in
    {critic = Layer.apply critic_linear ys; actor = Layer.apply actor_linear ys}

module Frame_stack : sig
  type t

  val create : unit -> t
  val update : t -> ?masks:Tensor.t -> Tensor.t -> Tensor.t
end = struct
  type t = Tensor.t

  let create () = Tensor.zeros [num_procs; num_stack; 84; 84] ~kind:Float

  let update t ?masks img =
    Option.iter masks ~f:(fun masks ->
        Tensor.(t *= view masks ~size:[num_procs; 1; 1; 1]) );
    let stack_slice i = Tensor.narrow t ~dim:1 ~start:i ~length:1 in
    for frame_index = 1 to num_stack - 1 do
      Tensor.copy_ (stack_slice (frame_index - 1)) ~src:(stack_slice frame_index)
    done;
    Tensor.copy_ (stack_slice (num_stack - 1)) ~src:img;
    t
end

let set tensor i src = Tensor.copy_ (Tensor.get tensor i) ~src

let () =
  let module E = Vec_env_gym_pyml in
  let envs = E.create "PongNoFrameskip-v4" ~num_processes:num_procs in
  Stdio.printf "Action space: %d\n%!" (E.action_space envs);
  let vs = Var_store.create ~name:"a2c" () in
  let model = model vs ~actions:2 in
  let optimizer = Optimizer.rmsprop vs ~learning_rate:7e-4 ~eps:1e-5 ~alpha:0.99 in
  let frame_stack = Frame_stack.create () in
  let obs = E.reset envs in
  Tensor.print_shape obs ~name:"obs";
  ignore (Frame_stack.update frame_stack obs : Tensor.t);
  let s_states =
    Tensor.zeros [num_steps + 1; num_procs; num_stack; 84; 84] ~kind:Float
  in
  let s_rewards = Tensor.zeros [num_steps; num_procs] in
  let s_actions = Tensor.zeros [num_steps; num_procs] ~kind:Int64 in
  let s_masks = Tensor.zeros [num_steps; num_procs] in
  for _index = 1 to updates do
    for s = 0 to num_steps - 1 do
      let {actor; critic = _} =
        Tensor.no_grad (fun () -> model (Tensor.get s_states s))
      in
      let probs = Tensor.softmax actor ~dim:(-1) in
      let actions =
        Tensor.multinomial probs ~num_samples:1 ~replacement:true
        |> Tensor.squeeze1 ~dim:(-1)
      in
      set s_actions s actions;
      let actions = Tensor.to_int1_exn actions |> Array.to_list in
      let {E.obs; reward; is_done} = E.step envs ~actions in
      let masks = Tensor.(f 1. - is_done) in
      let obs = Frame_stack.update frame_stack obs ~masks in
      set s_states (s + 1) obs;
      set s_rewards s reward;
      set s_masks s masks
    done;
    let s_returns =
      let r = Tensor.zeros [num_steps + 1; num_procs] in
      let {actor = _; critic} =
        Tensor.no_grad (fun () -> model (Tensor.get s_states (-1)))
      in
      set r (-1) (Tensor.view critic ~size:[num_procs]);
      for s = num_steps - 1 downto 0 do
        set r s Tensor.((get r Int.(s + 1) * f 0.99 * get s_masks s) + get s_rewards s)
      done;
      r
    in
    let {actor; critic} =
      model
        ( Tensor.narrow s_states ~dim:0 ~start:0 ~length:num_steps
        |> Tensor.view ~size:[num_steps * num_procs; num_stack; 84; 84] )
    in
    let critic = Tensor.view critic ~size:[num_steps; num_procs] in
    let actor = Tensor.view actor ~size:[num_steps; num_procs; -1] in
    let log_probs = Tensor.log_softmax actor ~dim:(-1) in
    let probs = Tensor.softmax actor ~dim:(-1) in
    let action_log_probs =
      Tensor.gather log_probs ~dim:2 ~index:(Tensor.unsqueeze s_actions ~dim:(-1))
      |> Tensor.squeeze1 ~dim:(-1)
    in
    let dist_entropy =
      Tensor.sum2 Tensor.(f 0. - (log_probs * probs)) ~dim:[-1] ~keepdim:true
      |> Tensor.mean
    in
    let advantages =
      Tensor.(narrow s_returns ~dim:0 ~start:0 ~length:num_steps - critic)
    in
    let value_loss = Tensor.(advantages * advantages) |> Tensor.mean in
    let action_loss =
      Tensor.(f 0. - (detach advantages * action_log_probs)) |> Tensor.mean
    in
    let loss = Tensor.((value_loss * f 0.05) + action_loss - (f 0.01 * dist_entropy)) in
    Optimizer.backward_step optimizer ~loss ~clip_grad_norm2:0.5;
    Caml.Gc.full_major ();
    set s_states 0 (Tensor.get s_states (-1))
  done
