open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let zeros =
      foreign "at_zeros"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> int      (* kind *)
        @-> returning t)

    let ones =
      foreign "at_ones"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> int      (* kind *)
        @-> returning t)

    let rand =
      foreign "at_rand"
        (   ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let reshape =
      foreign "at_reshape"
        (   t
        @-> ptr int  (* dims *)
        @-> int      (* num dims *)
        @-> returning t)

    let add = foreign "at_add" (t @-> t @-> returning t)
    let mul = foreign "at_mul" (t @-> t @-> returning t)
    let sub = foreign "at_sub" (t @-> t @-> returning t)
    let div = foreign "at_div" (t @-> t @-> returning t)
    let pow = foreign "at_pow" (t @-> t @-> returning t)
    let matmul = foreign "at_matmul" (t @-> t @-> returning t)

    let print =
      foreign "at_print"
        (t
        @-> returning void)

    let free =
      foreign "at_free"
        (t
        @-> returning void)
  end
end
