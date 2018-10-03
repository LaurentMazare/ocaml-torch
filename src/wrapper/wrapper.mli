module Kind : sig
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
end

module Tensor : sig
  type t
  val zeros : ?kind:Kind.t -> int list -> t
  val ones : ?kind:Kind.t -> int list -> t
  val rand : int list -> t

  val shape : t -> int list
  val kind : t -> Kind.t
  val reshape : t -> int list -> t

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val pow : t -> t -> t
  val matmul : t -> t -> t

  val print : t -> unit
end
