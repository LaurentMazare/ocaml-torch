open Torch_core.Wrapper

let () =
  let tensor1 = Tensor.zeros [1, 2] in
  let tensor2 = Tensor.ones [1, 2] in
  Tensor.print tensor1;
  Tensor.print tensor2
