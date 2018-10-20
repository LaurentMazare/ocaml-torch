open Base
include Torch_core.Wrapper.Optimizer

let adam tensors ~learning_rate =
  adam (List.map tensors ~f:Tensor.to_ptr) ~learning_rate

let sgd
    ?(momentum=0.)
    ?(dampening=0.)
    ?(weight_decay=0.)
    ?(nesterov=false)
    tensors
    ~learning_rate
  =
  sgd (List.map tensors ~f:Tensor.to_ptr)
    ~learning_rate
    ~momentum
    ~dampening
    ~weight_decay
    ~nesterov

let backward_step t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step t
