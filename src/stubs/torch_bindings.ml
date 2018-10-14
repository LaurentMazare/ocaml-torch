open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let float_vec =
      foreign "at_float_vec"
        (   ptr double (* values *)
        @-> int        (* num values *)
        @-> int        (* kind *)
        @-> returning t)

    let int_vec =
      foreign "at_int_vec"
        (   ptr int64_t (* values *)
        @-> int         (* num values *)
        @-> int         (* kind *)
        @-> returning t)

    let num_dims = foreign "at_dim" (t @-> returning int)

    let shape =
      foreign "at_shape"
        (   t
        @-> ptr int  (* dims *)
        @-> returning void)

    let scalar_type = foreign "at_scalar_type" (t @-> returning int)

    let backward = foreign "at_backward" (t @-> returning void)
    let requires_grad =
      foreign "at_requires_grad" (t @-> returning int)

    let get = foreign "at_get" (t @-> int @-> returning t)
    let select = foreign "at_select" (t @-> int @-> int @-> returning t)
    let double_value = foreign "at_double_value" (t @-> returning float)
    let int64_value = foreign "at_int64_value" (t @-> returning int64_t)
    let fill_double = foreign "at_fill_double" (t @-> float @-> returning void)
    let fill_int64 = foreign "at_fill_int64" (t @-> int64_t @-> returning void)

    let set_double2 =
      foreign "at_set_double2" (t @-> int @-> int @-> float @-> returning void)

    let print = foreign "at_print" (t @-> returning void)
    let free = foreign "at_free" (t @-> returning void)
  end

  module Serialize = struct
    let t = Tensor.t
    let save = foreign "at_save" (t @-> string @-> returning void)
    let load = foreign "at_load" (string @-> returning t)
    let save_multi =
      foreign "at_save_multi" (ptr t @-> ptr string @-> int @-> string @-> returning void)
    let load_multi =
      foreign "at_load_multi" (ptr t @-> ptr string @-> int @-> string @-> returning void)
    let load_multi_ =
      foreign "at_load_multi_" (ptr t @-> ptr string @-> int @-> string @-> returning void)
  end

  module Optimizer = struct
    type t = unit ptr
    let t : t typ = ptr void

    let adam =
      foreign "ato_adam" (ptr Tensor.t @-> int @-> float @-> returning t)
    let zero_grad = foreign "ato_zero_grad" (t @-> returning void)
    let step = foreign "ato_step" (t @-> returning void)
    let free = foreign "ato_free" (t @-> returning void)
  end

  module TensorG = Torch_bindings_generated.C(F)
end
