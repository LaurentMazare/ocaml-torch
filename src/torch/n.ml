open Base

type t = string list

let of_list = Fn.id
let to_string t = List.rev t |> String.concat ~sep:"."
let (/) t p = p :: t
