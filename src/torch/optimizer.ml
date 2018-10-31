open! Base
module Optimizer = Torch_core.Wrapper.Optimizer

type t =
  { optimizer : Optimizer.t
  ; vs : Var_store.t
  ; mutable parameters_in_optimizer : int
  }

let add_missing_parameters t =
  let tensors = Var_store.trainable_vars t.vs in
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

let clip_grad_norm2_ t ~max_norm2 =
  let total_norm =
    Var_store.trainable_vars t.vs
    |> List.fold ~init:0. ~f:(fun acc tensor ->
      let grad = Tensor.grad tensor in
      let grad_norm =
        if Tensor.defined grad
        then Tensor.norm1 grad |> Tensor.float_value
        else 0.
      in
      acc +. grad_norm)
    |> Float.sqrt
  in
  let clip_coef = max_norm2 /. (1e-6 +. total_norm) in
  if Float.(<) clip_coef 1.
  then begin
    let clip_coef = Tensor.f clip_coef in
    Var_store.trainable_vars t.vs
    |> List.iter ~f:(fun tensor ->
      let grad = Tensor.grad tensor in
      if Tensor.defined grad
      then ignore (Tensor.mul_ grad clip_coef : Tensor.t))
  end

let zero_grad t =
  add_missing_parameters t;
  Optimizer.zero_grad t.optimizer

let step ?clip_grad_norm2 t =
  add_missing_parameters t;
  Option.iter clip_grad_norm2 ~f:(fun max_norm2 ->
    clip_grad_norm2_ t ~max_norm2);
  Optimizer.step t.optimizer

let backward_step ?clip_grad_norm2 t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step ?clip_grad_norm2 t

let set_learning_rate t ~learning_rate =
  Optimizer.set_learning_rate t.optimizer learning_rate
