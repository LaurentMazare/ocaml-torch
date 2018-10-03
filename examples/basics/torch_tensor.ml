open Torch_core.Wrapper

let () =
  let tensor1 = Tensor.rand [4; 2] in
  let tensor2 = Tensor.ones [4; 2] in
  let sum = Tensor.add tensor1 tensor2 in
  Tensor.print tensor1;
  Tensor.print tensor2;
  Tensor.print sum;
  Tensor.print (Tensor.reshape sum [8])
