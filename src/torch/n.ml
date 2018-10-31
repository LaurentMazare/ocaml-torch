open Base

type t = string list

let of_list = Fn.id
let to_string t = List.rev t |> String.concat ~sep:"."
let (/) t p = p :: t

let counter =
  let v = ref 0 in
  fun () ->
    Int.incr v;
    !v

let default t_option base =
  match t_option with
  | Some t -> t
  | None -> [ Printf.sprintf "%s__%d" base (counter ()) ]
