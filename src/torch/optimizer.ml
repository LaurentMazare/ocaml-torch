open! Base
open Torch_core.Wrapper

type t =
  { optimizer : Optimizer.t
  ; vs : Var_store.t
  ; mutable parameters_in_optimizer : int
  }

let add_missing_parameters t =
  let tensors = Var_store.vars t.vs `trainable in
  let missing = List.length tensors - t.parameters_in_optimizer in
  if missing > 0
  then begin
    (* This only works as [Var_store.vars] returns tensors in
       reverse order of them being added to the store. *)
    let to_add = List.take tensors missing in
    Optimizer.add_parameters t.optimizer to_add;
    t.parameters_in_optimizer <- t.parameters_in_optimizer + List.length to_add
  end

let create optimizer ~vs =
  let t =
    { optimizer
    ; vs
    ; parameters_in_optimizer = 0
    }
  in
  add_missing_parameters t;
  t

let adam vs ~learning_rate =
  Optimizer.adam ~learning_rate
  |> create ~vs

let sgd
    ?(momentum=0.)
    ?(dampening=0.)
    ?(weight_decay=0.)
    ?(nesterov=false)
    vs
    ~learning_rate
  =
  Optimizer.sgd
    ~learning_rate
    ~momentum
    ~dampening
    ~weight_decay
    ~nesterov
  |> create ~vs

let backward_step t ~loss =
  add_missing_parameters t;
  Optimizer.zero_grad t.optimizer;
  Tensor.backward loss;
  Optimizer.step t.optimizer

let zero_grad t =
  add_missing_parameters t;
  Optimizer.zero_grad t.optimizer

let step t =
  add_missing_parameters t;
  Optimizer.step t.optimizer

let set_learning_rate t ~learning_rate =
  Optimizer.set_learning_rate t.optimizer learning_rate
