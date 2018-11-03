open Base
open Torch

let print_one filename =
  Stdio.printf "%s:\n" filename;
  Serialize.load_all ~filename
  |> List.iter ~f:(fun (tensor_name, tensor) ->
    let shape =
      Tensor.shape tensor
      |> List.map ~f:Int.to_string
      |> String.concat ~sep:", "
    in
    Stdio.printf "  %s (%s)\n" tensor_name shape)

let () =
  match Array.to_list Sys.argv with
  | [] ->
    Stdio.printf "Usage: %s file1.npz file2.npz...\n" Sys.argv.(0)
  | _ :: argv -> List.iter argv ~f:print_one

