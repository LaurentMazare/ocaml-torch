type _ t =
  | Uint8 : [ `u8 ] t
  | Int8 : [ `i8 ] t
  | Int16 : [ `i16 ] t
  | Int : [ `i32 ] t
  | Int64 : [ `i64 ] t
  | Half : [ `f16 ] t
  | Float : [ `f32 ] t
  | Double : [ `f64 ] t
  | ComplexHalf : [ `c16 ] t
  | ComplexFloat : [ `c32 ] t
  | ComplexDouble : [ `c64 ] t
  | Bool : [ `bool ] t

type packed = T : _ t -> packed

(* Hardcoded, should match ScalarType.h *)
let to_int : type a. a t -> int = function
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

let packed_to_int (T t) = to_int t

let of_int_exn = function
  | 0 -> T Uint8
  | 1 -> T Int8
  | 2 -> T Int16
  | 3 -> T Int
  | 4 -> T Int64
  | 5 -> T Half
  | 6 -> T Float
  | 7 -> T Double
  | 8 -> T ComplexHalf
  | 9 -> T ComplexFloat
  | 10 -> T ComplexDouble
  | 11 -> T Bool
  | d -> failwith (Printf.sprintf "unexpected kind %d" d)

let (<>) packed1 packed2 =
  packed_to_int packed1 <> packed_to_int packed2
