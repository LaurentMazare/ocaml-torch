type t =
  | Uint8
  | Int8
  | Int16
  | Int
  | Int64
  | Half
  | Float
  | Double
  | ComplexHalf
  | ComplexFloat
  | ComplexDouble
  | Bool

(* Hardcoded, should match ScalarType.h *)
let to_int = function
  | Uint8 -> 0
  | Int8 -> 1
  | Int16 -> 2
  | Int -> 3
  | Int64 -> 4
  | Half -> 5
  | Float -> 6
  | Double -> 7
  | ComplexHalf -> 8
  | ComplexFloat -> 9
  | ComplexDouble -> 10
  | Bool -> 11

let of_int_exn = function
  | 0 -> Uint8
  | 1 -> Int8
  | 2 -> Int16
  | 3 -> Int
  | 4 -> Int64
  | 5 -> Half
  | 6 -> Float
  | 7 -> Double
  | 8 -> ComplexHalf
  | 9 -> ComplexFloat
  | 10 -> ComplexDouble
  | 11 -> Bool
  | d -> failwith (Printf.sprintf "unexpected kind %d" d)

let ( = ) = Stdlib.( = )
let ( <> ) = Stdlib.( <> )
