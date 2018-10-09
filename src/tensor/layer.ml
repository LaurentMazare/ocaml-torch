open Base

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu
  | Sigmoid

module Linear = struct
  type t =
    { output_dim : int
    ; mutable w : Tensor.t option
    ; mutable b : Tensor.t option
    }

  let vars { w; b; output_dim = _ } =
    [ Option.value_exn w; Option.value_exn b ]

  let create output_dim =
    { output_dim
    ; w = None
    ; b = None
    }

  let apply ?activation ?(use_bias=true) t xs =
    let last_xs_dim = Tensor.shape xs |> List.last_exn in
    let w =
      match t.w with
      | Some w -> w
      | None ->
        let w =
          Tensor.randn [ last_xs_dim; t.output_dim ] ~scale:0.1 ~requires_grad:true
        in
        t.w <- Some w;
        w
    in
    let b =
      match t.b with
      | Some b -> b
      | None ->
        let b = Tensor.zeros [ t.output_dim ] ~requires_grad:true in
        t.b <- Some b;
        b
    in
    let ys = if use_bias then Tensor.(mm xs w + b) else Tensor.(mm xs w) in
    match activation with
    | Some Relu -> Tensor.relu ys
    | Some Softmax -> Tensor.softmax ys
    | Some Tanh -> Tensor.tanh ys
    | Some Sigmoid -> Tensor.sigmoid ys
    | Some Leaky_relu -> Tensor.leaky_relu ys
    | None -> ys
end
