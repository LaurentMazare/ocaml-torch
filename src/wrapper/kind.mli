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

val to_int : t -> int
val of_int_exn : int -> t
val ( = ) : t -> t -> bool
val ( <> ) : t -> t -> bool
