open Base
include Torch_core.Wrapper.Optimizer

let adam vs ~learning_rate =
  let tensors = Var_store.vars vs `trainable |> List.map ~f:Tensor.to_ptr in
  adam tensors ~learning_rate

let sgd
    ?(momentum=0.)
    ?(dampening=0.)
    ?(weight_decay=0.)
    ?(nesterov=false)
    vs
    ~learning_rate
  =
  let tensors = Var_store.vars vs `trainable |> List.map ~f:Tensor.to_ptr in
  sgd tensors
    ~learning_rate
    ~momentum
    ~dampening
    ~weight_decay
    ~nesterov

let backward_step t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step t

let set_learning_rate t ~learning_rate = set_learning_rate t learning_rate
