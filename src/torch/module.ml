open Base
include Torch_core.Wrapper.Module

let forward_ t ivalues = List.map ~f:Ivalue.to_raw ivalues |> forward_ t |> Ivalue.of_raw
