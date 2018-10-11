include Torch_core.Wrapper.Optimizer

let backward_step t ~loss =
  zero_grad t;
  Tensor.backward loss;
  step t
