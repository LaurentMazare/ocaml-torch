open Base
open Torch

let batch_size = 32
let updates = 1_000

module Model : sig
  type t
  val create : Var_store.t -> input_dim:int -> int -> t
  val logits_and_value : t -> Tensor.t -> Tensor.t * Tensor.t
  val action_and_value : t -> Tensor.t -> num_samples:int -> Tensor.t * Tensor.t
end = struct
  type t =
    { linear1 : Layer.t
    ; linear2 : Layer.t
    ; value : Layer.t
    ; logits : Layer.t
    }

  let create vs ~input_dim actions_dim =
    { linear1 = Layer.linear vs ~input_dim 128
    ; linear2 = Layer.linear vs ~input_dim:128 128
    ; value = Layer.linear vs ~input_dim:128 1
    ; logits = Layer.linear vs ~input_dim:128 actions_dim
    }

  let logits_and_value t xs =
    let logits = Layer.apply t.linear1 xs |> Tensor.relu |> Layer.apply t.logits in
    let value = Layer.apply t.linear2 xs |> Tensor.relu |> Layer.apply t.value in
    logits, value

  let action_and_value t xs ~num_samples =
    let logits, value = logits_and_value t xs in
    let action =
      Tensor.softmax logits ~dim:1
      |> Tensor.multinomial ~num_samples ~replacement:true
    in
    action, value
end

module A2CAgent : sig
  type t
  val create : state_dim:int -> actions:int -> t
  val train : t -> unit
end = struct
  type t =
    { model : Model.t
    ; actions : int
    ; batch_size : int
    ; gamma : float
    ; value : float
    ; entropy : float
    ; optimizer : Optimizer.t
    }

  let create ~state_dim ~actions =
    let vs = Var_store.create ~name:"a2c" () in
    let model = Model.create vs ~input_dim:state_dim actions in
    let optimizer = Optimizer.adam vs ~learning_rate:1e-3 in
    { model
    ; actions
    ; batch_size = 32
    ; gamma = 0.99
    ; value = 0.5
    ; entropy = 1e-4
    ; optimizer
    }

  let train t =
    let returns, value, policy_loss, entropy_loss = failwith "TODO" in
    let value_loss = Tensor.mse_loss returns value in
    let loss = Tensor.(f t.value * value_loss + policy_loss - f t.entropy * entropy_loss) in
    Optimizer.backward_step t.optimizer ~loss
end

let () =
  let module E = Env_gym_pyml in
  let env = E.create "CartPole-v1" in
  let agent = A2CAgent.create ~state_dim:4 ~actions:2 in
  for _index = 1 to updates do
    for _batch_index = 1 to batch_size do
      ()
    done;
    A2CAgent.train agent;
  done;
  ignore (Model.action_and_value, Model.logits_and_value, agent, env)
