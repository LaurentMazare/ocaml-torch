open Torch

let tensor_ops () =
  let tensor1 = Tensor.rand [4; 2] in
  let tensor2 = Tensor.ones [4; 2] in
  let sum = Tensor.(+) tensor1 tensor2 in
  Tensor.fill_float (Tensor.get tensor1 1) 42.0;
  Tensor.print tensor1;
  Tensor.print tensor2;
  Tensor.print sum;
  Tensor.print (Tensor.reshape sum ~shape:[8]);
  let v = Tensor.get sum 3 |> fun t -> Tensor.get t 0 |> Tensor.float_value in
  Printf.printf "sum[3] = %f\n%!" v

let backward_pass () =
  let x = Tensor.float_vec [-1.0; 0.0; 1.0; 2.0] |> Tensor.set_requires_grad ~r:true in
  let square = Tensor.(x * x) in
  Tensor.backward square;
  Tensor.print square;
  Tensor.print (Tensor.grad x)

let bigarray () =
  let ba = Bigarray.Array2.create Float32 C_layout 5 2 in
  let ba2 = Bigarray.Array2.create Float32 C_layout 5 2 in
  Bigarray.Array2.fill ba 1337.;
  Bigarray.Array2.fill ba2 1234.;
  ba.{1, 1} <- 42.0;
  let t = Bigarray.genarray_of_array2 ba |> Tensor.of_bigarray in
  Tensor.print t;
  Tensor.copy_to_bigarray t (Bigarray.genarray_of_array2 ba2);
  ba2.{1, 1} <- 42.5;
  Bigarray.genarray_of_array2 ba2 |> Tensor.of_bigarray |> Tensor.print

let () =
  bigarray ();
  tensor_ops ();
  backward_pass ();
  Stdio.printf "done\n%!"
