type t =
  | Cpu
  | Cuda of int

(* Hardcoded, should match torch_api.cpp *)
let to_int = function
  | Cpu -> -1
  | Cuda i ->
    if i < 0 then Printf.sprintf "negative index for cuda device" |> failwith;
    i

let of_int i = if i < 0 then Cpu else Cuda i
