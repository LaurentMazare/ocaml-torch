open! Ctypes

module C = Torch_bindings.C(Torch_generated)

module Tensor = struct
  open! C.Tensor
  type nonrec t = t

  let zeros dims =
    let dim_array =
      CArray.of_list int dims
      |> CArray.start
    in
    zeros dim_array (List.length dims)

  let ones dims =
    let dim_array =
      CArray.of_list int dims
      |> CArray.start
    in
    ones dim_array (List.length dims)

  let add = add

  let print = print
end
