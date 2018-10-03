open! Ctypes

module C = Torch_bindings.C(Torch_generated)

module Tensor = struct
  open! C.Tensor
  type nonrec t = t

  let zeros dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = zeros dim_array (List.length dims) in
    Gc.finalise free tensor;
    tensor

  let ones dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = ones dim_array (List.length dims) in
    Gc.finalise free tensor;
    tensor

  let rand dims =
    let dim_array = CArray.of_list int dims |> CArray.start in
    let tensor = rand dim_array (List.length dims) in
    Gc.finalise free tensor;
    tensor

  let add x y =
    let tensor = add x y in
    Gc.finalise free tensor;
    tensor

  let print = print
end
