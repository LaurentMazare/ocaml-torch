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

let u8 = Uint8
let i8 = Int8
let i16 = Int16
let i32 = Int
let i64 = Int64
let f16 = Half
let f32 = Float
let f64 = Double
let c16 = ComplexHalf
let c32 = ComplexFloat
let c64 = ComplexDouble
let bool = Bool

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

let to_string : type a. a t -> string = function
  | Uint8 -> "u8"
  | Int8 -> "i8"
  | Int16 -> "i16"
  | Int -> "i32"
  | Int64 -> "i64"
  | Half -> "f16"
  | Float -> "f32"
  | Double -> "f64"
  | ComplexHalf -> "c16"
  | ComplexFloat -> "c32"
  | ComplexDouble -> "c64"
  | Bool -> "bool"

let packed_to_string (T t) = to_string t

let (<>) packed1 packed2 =
  packed_to_int packed1 <> packed_to_int packed2
