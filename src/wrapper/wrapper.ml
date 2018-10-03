open! Ctypes

module C = Torch_bindings.C(Torch_generated)

module Tensor = struct
  open! C.Tensor
  type nonrec t = t
  let zeros _dims = zeros 0
  let ones _dims = ones 0
  let add = add
  let print = print
end
