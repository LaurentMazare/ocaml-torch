open Torch

let () =
  Stdio.printf "cuda available: %b\n%!" (Cuda.is_available ());
  Stdio.printf "cudnn available: %b\n%!" (Cuda.cudnn_is_available ());
  let t = Tensor.(f 41. + f 1.) in
  Tensor.print t
