open Torch_core.Wrapper

let tensor_ops () =
  let tensor1 = Tensor.rand [4; 2] in
  let tensor2 = Tensor.ones [4; 2] in
  let sum = Tensor.add tensor1 tensor2 in
  Tensor.print tensor1;
  Tensor.print tensor2;
  Tensor.print sum;
  Tensor.print (Tensor.reshape sum [8]);
  let v = Tensor.get sum 3 |> fun t -> Tensor.get t 0 |> Tensor.float_value in
  Printf.printf "sum[3] = %f\n%!" v

let backward_pass () =
  let x = Tensor.float_vec [-1.0; 0.0; 1.0; 2.0] |> Tensor.set_requires_grad ~b:true in
  let square = Tensor.mul x x in
  Tensor.backward square;
  Tensor.print square;
  Tensor.print (Tensor.grad x)

let () =
  tensor_ops ();
  backward_pass ()
