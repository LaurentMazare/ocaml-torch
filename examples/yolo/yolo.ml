open Base
open Torch
open Torch_vision
let failwithf = Printf.failwithf
let config_filename = "examples/yolo/yolo-v3.cfg"

module Darknet = struct
  type block =
    { block_type : string
    ; parameters : (string, string, String.comparator_witness) Map.t
    }

  type t =
    { blocks : block list
    ; parameters : (string, string, String.comparator_witness) Map.t
    }

  let parse_config filename =
    let blocks =
      Stdio.In_channel.read_lines filename
      |> List.filter_map ~f:(fun line ->
          let line = String.strip line in
          if String.is_empty line || Char.(=) line.[0] '#'
          then None
          else Some line)
      |> List.group ~break:(fun _ line -> Char.(=) line.[0] '[')
      |> List.map ~f:(function
        | block_type :: paramaters ->
            let block_type =
              match String.chop_prefix block_type ~prefix:"[" with
              | None -> failwithf "block-type does not start with [: %s" block_type ()
              | Some block_type ->
                  match String.chop_suffix block_type ~suffix:"]" with
                  | None -> failwithf "block-type does not end with ]: %s" block_type ()
                  | Some block_type -> block_type
            in
            let parameters = List.map paramaters ~f:(fun line ->
              match String.split line ~on:'=' with
              | [ lhs; rhs ] -> lhs, rhs
              | _ -> failwithf "parameter line does not contain exactly one equal: %s" line ())
            in
            let parameters =
              match Map.of_alist (module String) parameters with
              | `Duplicate_key key -> failwithf "multiple %s key for %s" key block_type ()
              | `Ok parameters -> parameters
            in
            { block_type; parameters }
        | _ -> assert false)
    in
    match blocks with
    | { block_type = "net"; parameters } :: blocks -> { blocks; parameters }
    | _ -> failwith "expected the first block to start with [net]"

  let build_model blocks =
    ignore blocks;
    failwith "TODO"
end

let () =
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s yolo-v3.ot input.png" Sys.argv.(0) ();
  let vs = Var_store.create ~name:"rn" ~device:Cpu () in
  let darknet = Darknet.parse_config config_filename in
  Stdio.printf "%d blocks in %s\n%!" (List.length darknet.blocks) config_filename;
  let model = Darknet.build_model darknet in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  let image = Imagenet.load_image Sys.argv.(2) in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  let probabilities =
    Layer.apply_ model image ~is_training:false |> Tensor.softmax ~dim:(-1)
  in
  Imagenet.Classes.top probabilities ~k:5
  |> List.iter ~f:(fun (name, probability) ->
      Stdio.printf "%s: %.2f%%\n%!" name (100. *. probability) )

