open Base

type t = string list

let of_list = Fn.id
let to_string t = List.rev t |> String.concat ~sep:"."
let of_string_parts str = String.split str ~on:'.' |> List.rev
let ( / ) t p = p :: t
let root = []
