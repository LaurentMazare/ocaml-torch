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

val u8 : [ `u8 ] t
val i8 : [ `i8 ] t
val i16 : [ `i16 ] t
val i32 : [ `i32 ] t
val i64 : [ `i64 ] t
val f16 : [ `f16 ] t
val f32 : [ `f32 ] t
val f64 : [ `f64 ] t
val c16 : [ `c16 ] t
val c32 : [ `c32 ] t
val c64 : [ `c64 ] t
val bool : [ `bool ] t

type packed = T : _ t -> packed

val to_int : _ t -> int
val packed_to_int : packed -> int
val of_int_exn : int -> packed
val (<>) : packed -> packed -> bool
