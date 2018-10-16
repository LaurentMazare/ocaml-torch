open Base
include Torch_core.Wrapper.Optimizer

let adam tensors ~learning_rate =
  adam (List.map tensors ~f:Tensor.to_ptr) ~learning_rate

let backward_step t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step t
