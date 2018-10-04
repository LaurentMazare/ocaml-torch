open Torch_tensor

let () =
  let mnist = Mnist_helper.read_files () in
  ignore mnist
