open! Base

module Var_store = struct
  type t = Tensor.t list ref
  let create () = ref []
  let add_vars t ~vars = t := vars @ !t
  let vars t = !t
end

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

module Linear = struct
  type t =
    { w : Tensor.t
    ; b : Tensor.t
    }

  let create vs ~input_dim output_dim =
    let w = Tensor.randn [ input_dim; output_dim ] ~scale:0.1 ~requires_grad:true in
    let b = Tensor.zeros [ output_dim ] ~requires_grad:true in
    Var_store.add_vars vs ~vars:[w; b];
    { w; b }

  let apply ?activation ?(use_bias=true) t xs =
    let ys = if use_bias then Tensor.(mm xs t.w + t.b) else Tensor.(mm xs t.w) in
    match activation with
    | Some Relu -> Tensor.relu ys
    | Some Softmax -> Tensor.softmax ys
    | Some Tanh -> Tensor.tanh ys
    | Some Sigmoid -> Tensor.sigmoid ys
    | Some Leaky_relu -> Tensor.leaky_relu ys
    | None -> ys
end

module Conv2D = struct
  type t =
    { w : Tensor.t
    ; b : Tensor.t
    ; stride : int * int
    ; padding : int * int
    }

  let create vs ~ksize:(k1, k2) ~stride ?(padding=0, 0) ~input_dim output_dim =
    let w =
      Tensor.randn [ output_dim; input_dim; k1; k2 ] ~scale:0.1 ~requires_grad:true
    in
    let b = Tensor.zeros [ output_dim ] ~requires_grad:true in
    Var_store.add_vars vs ~vars:[w; b];
    { w; b; stride; padding }

  let apply ?activation t xs =
    let ys =
      Tensor.conv2d xs t.w t.b
        ~padding:t.padding
        ~stride:t.stride
    in
    match activation with
    | Some Relu -> Tensor.relu ys
    | Some Softmax -> Tensor.softmax ys
    | Some Tanh -> Tensor.tanh ys
    | Some Sigmoid -> Tensor.sigmoid ys
    | Some Leaky_relu -> Tensor.leaky_relu ys
    | None -> ys
end
