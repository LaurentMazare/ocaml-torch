open! Base

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

  let vars { w; b } = [ w; b ]

  let create ~input_dim output_dim =
    { w = Tensor.randn [ input_dim; output_dim ] ~scale:0.1 ~requires_grad:true
    ; b = Tensor.zeros [ output_dim ] ~requires_grad:true
    }

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
