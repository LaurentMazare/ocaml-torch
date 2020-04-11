open Base
module I = Torch_core.Wrapper.Ivalue

type raw = I.t

type t =
  | None
  | Bool of bool
  | Tensor of Tensor.t
  | Int of int
  | Double of float
  | Tuple of t list
  | String of string

let rec to_string = function
  | None -> "none"
  | Bool b -> Bool.to_string b
  | Tensor t -> Tensor.shape_str t
  | Int i -> Int.to_string i
  | Double f -> Float.to_string f
  | Tuple ts ->
    List.map ts ~f:to_string |> String.concat ~sep:", " |> Printf.sprintf "(%s)"
  | String s -> Printf.sprintf "\"%s\"" s

let rec to_raw = function
  | None -> I.none ()
  | Bool bool -> I.bool bool
  | Tensor tensor -> I.tensor tensor
  | Int int -> I.int64 (Int64.of_int int)
  | Double double -> I.double double
  | Tuple tuple -> I.tuple (List.map ~f:to_raw tuple)
  | String string -> I.string string

let rec of_raw ivalue =
  match I.tag ivalue with
  | Tensor -> Tensor (I.to_tensor ivalue)
  | Int -> Int (I.to_int64 ivalue |> Int64.to_int_exn)
  | Double -> Double (I.to_double ivalue)
  | Tuple -> Tuple (I.to_tuple ivalue |> List.map ~f:of_raw)
  | None -> None
  | Bool -> Bool (I.to_bool ivalue)
  | String -> String (I.to_string ivalue)
  | _ -> failwith "unsupported tag"
