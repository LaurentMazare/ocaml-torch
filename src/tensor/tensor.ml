include Base
include Torch_core.Wrapper.Tensor

(* TODO: implement these in a single C call rather than chaining [get]. *)
let set_float1 t i value = fill_float (get t i) value
let set_int2 t i j value = fill_int (get (get t i) j) value
let set_int1 t i value = fill_int (get t i) value

let get_float2 t i j = float_value (get (get t i) j)
let get_float1 t i = float_value (get t i)
let get_int2 t i j = int_value (get (get t i) j)
let get_int1 t i = int_value (get t i)

let no_grad t ~f =
  if requires_grad t
  then
    let t = set_requires_grad t ~b:false in
    let result = f t in
    ignore (set_requires_grad t ~b:true : t);
    result
  else f t

let zeros ?(requires_grad = false) ?kind dims =
  let t = zeros ?kind dims in
  if requires_grad
  then set_requires_grad t ~b:true
  else t

let f v = float_vec [ v ] |> reshape ~dims:[]
let mm = matmul

let (+) = add
let (-) = sub
let ( * ) = mul
let (/) = div

let (~-) = neg
let (-=) = sub_assign
let (=) = eq
