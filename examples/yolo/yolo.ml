(* The pre-trained weights can be downloaded here:
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot
*)
(* TODO: preserve the aspect ratio when loading the images. *)
(* TODO: draw bboxes on the initial images. *)
open Base
open Torch
open Torch_vision

let sprintf = Printf.sprintf
let failwithf = Printf.failwithf
let config_filename = "examples/yolo/yolo-v3.cfg"
let confidence_threshold = 0.5
let nms_threshold = 0.4
let classes = Coco_names.names

module Darknet = struct
  type block =
    { block_type : string
    ; parameters : (string, string, String.comparator_witness) Map.t
    }

  type t =
    { blocks : block list
    ; parameters : (string, string, String.comparator_witness) Map.t
    }

  let int_list_of_string str =
    String.split str ~on:',' |> List.map ~f:(fun i -> String.strip i |> Int.of_string)

  let parse_config filename =
    let blocks =
      Stdio.In_channel.read_lines filename
      |> List.filter_map ~f:(fun line ->
             let line = String.strip line in
             if String.is_empty line || Char.( = ) line.[0] '#' then None else Some line
         )
      |> List.group ~break:(fun _ line -> Char.( = ) line.[0] '[')
      |> List.map ~f:(function
             | block_type :: paramaters ->
               let block_type =
                 match String.chop_prefix block_type ~prefix:"[" with
                 | None -> failwithf "block-type does not start with [: %s" block_type ()
                 | Some block_type ->
                   (match String.chop_suffix block_type ~suffix:"]" with
                   | None -> failwithf "block-type does not end with ]: %s" block_type ()
                   | Some block_type -> block_type)
               in
               let parameters =
                 List.map paramaters ~f:(fun line ->
                     match String.split line ~on:'=' with
                     | [ lhs; rhs ] -> String.strip lhs, String.strip rhs
                     | _ ->
                       failwithf
                         "parameter line does not contain exactly one equal: %s"
                         line
                         ())
               in
               let parameters =
                 match Map.of_alist (module String) parameters with
                 | `Duplicate_key key ->
                   failwithf "multiple %s key for %s" key block_type ()
                 | `Ok parameters -> parameters
               in
               { block_type; parameters }
             | _ -> assert false)
    in
    match blocks with
    | { block_type = "net"; parameters } :: blocks -> { blocks; parameters }
    | _ -> failwith "expected the first block to start with [net]"

  let find_key key ~index ~parameters ~f =
    match Map.find parameters key with
    | None -> failwithf "cannot find key %s for block %d" key index ()
    | Some value ->
      (try f value with
      | _ -> failwithf "unable to convert '%s' (key %s, block %d)" value key index ())

  let convolutional vs ~index ~prev_channels ~parameters =
    let activation = find_key ~index ~parameters "activation" ~f:String.lowercase in
    let next_channels = find_key ~index ~parameters "filters" ~f:Int.of_string in
    let padding = find_key ~index ~parameters "pad" ~f:Int.of_string in
    let ksize = find_key ~index ~parameters "size" ~f:Int.of_string in
    let stride = find_key ~index ~parameters "stride" ~f:Int.of_string in
    let padding = if padding <> 0 then (ksize - 1) / 2 else 0 in
    let bn, use_bias =
      match Map.find parameters "batch_normalize" with
      | Some b when Int.of_string b <> 0 ->
        let bn =
          Layer.batch_norm2d Var_store.(vs / sprintf "batch_norm_%d" index) next_channels
        in
        bn, false
      | Some _ | None -> Layer.id_, true
    in
    let conv =
      Layer.conv2d_
        Var_store.(vs / sprintf "conv_%d" index)
        ~ksize
        ~stride
        ~padding
        ~input_dim:prev_channels
        ~use_bias
        next_channels
      |> Layer.with_training
    in
    let activation =
      match activation with
      | "leaky" ->
        Layer.of_fn (fun xs -> Tensor.(max xs (xs * f 0.1))) |> Layer.with_training
      | "linear" -> Layer.id_
      | activation -> failwithf "unsupported activation %s block %d" activation index ()
    in
    next_channels, `layers [ conv; bn; activation ]

  let upsample ~index:_ ~prev_channels ~parameters:_ =
    let layer =
      Layer.of_fn (fun xs ->
          let _n, _c, h, w = Tensor.shape4_exn xs in
          Tensor.upsample_nearest2d xs ~output_size:[ h * 2; w * 2 ])
      |> Layer.with_training
    in
    prev_channels, `layers [ layer ]

  let route ~index ~prevs ~parameters =
    let layers =
      find_key ~index ~parameters "layers" ~f:int_list_of_string
      |> List.map ~f:(fun i -> if i >= 0 then index - i else -i)
    in
    let channels =
      List.sum (module Int) layers ~f:(fun i -> List.nth_exn prevs (i - 1) |> fst)
    in
    channels, `route layers

  let shortcut ~index ~prev_channels ~parameters =
    let from = find_key ~index ~parameters "from" ~f:Int.of_string in
    prev_channels, `shortcut (if from >= 0 then index - from else -from)

  let yolo ~index ~prev_channels ~parameters =
    let anchors =
      find_key ~index ~parameters "anchors" ~f:int_list_of_string
      |> List.groupi ~break:(fun i _ _ -> i % 2 = 0)
      |> List.map ~f:(function
             | [ p; q ] -> p, q
             | _ -> failwithf "odd number of elements in mask at index %d" index ())
      |> Array.of_list
    in
    let anchors =
      find_key ~index ~parameters "mask" ~f:int_list_of_string
      |> List.map ~f:(fun i -> anchors.(i))
    in
    let classes = find_key ~index ~parameters "classes" ~f:Int.of_string in
    prev_channels, `yolo (classes, anchors)

  let width t = find_key "width" ~index:(-1) ~parameters:t.parameters ~f:Int.of_string
  let height t = find_key "height" ~index:(-1) ~parameters:t.parameters ~f:Int.of_string

  let slice_apply_and_set xs ~start ~length ~f =
    let slice = Tensor.narrow xs ~dim:2 ~start ~length in
    Tensor.copy_ slice ~src:(f slice)

  let detect xs ~image_height ~anchors ~classes ~device =
    let bsize, _channels, height, _width = Tensor.shape4_exn xs in
    let stride = image_height / height in
    let grid_size = image_height / stride in
    let anchors =
      List.map anchors ~f:(fun (x, y) ->
          Float.(of_int x / of_int stride, of_int y / of_int stride))
    in
    let num_anchors = List.length anchors in
    let bbox_attrs = 5 + classes in
    let xs =
      Tensor.view xs ~size:[ bsize; bbox_attrs * num_anchors; grid_size * grid_size ]
      |> Tensor.transpose ~dim0:1 ~dim1:2
      |> Tensor.contiguous
      |> Tensor.view ~size:[ bsize; grid_size * grid_size * num_anchors; bbox_attrs ]
    in
    let grid = Tensor.arange ~end_:(Scalar.int grid_size) ~options:(Float, device) in
    let a = Tensor.repeat grid ~repeats:[ grid_size; 1 ] in
    let b = Tensor.tr a |> Tensor.contiguous in
    let x_offset = Tensor.view a ~size:[ -1; 1 ] in
    let y_offset = Tensor.view b ~size:[ -1; 1 ] in
    let xy_offset =
      Tensor.cat [ x_offset; y_offset ] ~dim:1
      |> Tensor.repeat ~repeats:[ 1; num_anchors ]
      |> Tensor.view ~size:[ -1; 2 ]
      |> Tensor.unsqueeze ~dim:0
    in
    slice_apply_and_set
      xs
      ~start:0
      ~length:2
      ~f:Tensor.(fun xs -> sigmoid xs + xy_offset);
    slice_apply_and_set xs ~start:4 ~length:(1 + classes) ~f:Tensor.sigmoid;
    let anchors =
      Array.of_list anchors
      |> Array.map ~f:(fun (x, y) -> [| x; y |])
      |> Tensor.of_float2
      |> Tensor.repeat ~repeats:[ grid_size * grid_size; 1 ]
      |> Tensor.unsqueeze ~dim:0
    in
    slice_apply_and_set xs ~start:2 ~length:2 ~f:Tensor.(fun xs -> exp xs * anchors);
    slice_apply_and_set
      xs
      ~start:0
      ~length:4
      ~f:Tensor.(fun xs -> xs * f (Float.of_int stride));
    xs

  let build_model vs t =
    let blocks =
      List.foldi t.blocks ~init:[] ~f:(fun index prevs block ->
          let vs = Var_store.(vs / Int.to_string index) in
          let prev_channels =
            match prevs with
            | [] -> 3
            | (channels, _) :: _ -> channels
          in
          let { block_type; parameters } = block in
          let block =
            match block_type with
            | "convolutional" -> convolutional vs ~index ~prev_channels ~parameters
            | "upsample" -> upsample ~index ~prev_channels ~parameters
            | "route" -> route ~index ~prevs ~parameters
            | "shortcut" -> shortcut ~index ~prev_channels ~parameters
            | "yolo" -> yolo ~index ~prev_channels ~parameters
            | bt -> failwithf "block-type %s is not implemented" bt ()
          in
          block :: prevs)
    in
    let blocks = List.rev blocks in
    let outputs = Hashtbl.create (module Int) in
    let image_height = height t in
    Layer.of_fn_ (fun xs ~is_training ->
        List.foldi
          blocks
          ~init:(xs, None)
          ~f:(fun index (xs, detections) (_channels, block) ->
            let ys, detections =
              match block with
              | `layers layers ->
                let ys =
                  List.fold layers ~init:xs ~f:(fun xs l ->
                      Layer.apply_ l xs ~is_training)
                in
                ys, detections
              | `route layers ->
                let ys =
                  List.map layers ~f:(fun i -> Hashtbl.find_exn outputs (index - i))
                  |> Tensor.cat ~dim:1
                in
                ys, detections
              | `shortcut from ->
                let ys =
                  Tensor.( + )
                    (Hashtbl.find_exn outputs (index - 1))
                    (Hashtbl.find_exn outputs (index - from))
                in
                ys, detections
              | `yolo (classes, anchors) ->
                let ys =
                  detect xs ~image_height ~anchors ~classes ~device:(Var_store.device vs)
                in
                let detections =
                  match detections with
                  | None -> ys
                  | Some detections -> Tensor.cat [ detections; ys ] ~dim:1
                in
                ys, Some detections
            in
            Hashtbl.add_exn outputs ~key:index ~data:ys;
            ys, detections)
        |> fun (_last, detections) -> Option.value_exn detections)
end

type bbox =
  { xmin : float
  ; ymin : float
  ; xmax : float
  ; ymax : float
  ; confidence : float
  ; class_index : int
  ; class_confidence : float
  }

let iou b1 b2 =
  let b1_area = (b1.xmax -. b1.xmin +. 1.) *. (b1.ymax -. b1.ymin +. 1.) in
  let b2_area = (b2.xmax -. b2.xmin +. 1.) *. (b2.ymax -. b2.ymin +. 1.) in
  let i_xmin = Float.max b1.xmin b2.xmin in
  let i_xmax = Float.min b1.xmax b2.xmax in
  let i_ymin = Float.max b1.ymin b2.ymin in
  let i_ymax = Float.min b1.ymax b2.ymax in
  let i_area =
    Float.max 0. (i_xmax -. i_xmin +. 1.) *. Float.max 0. (i_ymax -. i_ymin +. 1.)
  in
  i_area /. (b1_area +. b2_area -. i_area)

let colors =
  [| [| 0.5; 0.0; 0.5 |]
   ; [| 0.0; 0.5; 0.5 |]
   ; [| 0.5; 0.5; 0.0 |]
   ; [| 0.7; 0.3; 0.0 |]
   ; [| 0.7; 0.0; 0.3 |]
   ; [| 0.0; 0.7; 0.3 |]
   ; [| 0.3; 0.7; 0.0 |]
   ; [| 0.3; 0.0; 0.7 |]
   ; [| 0.0; 0.3; 0.7 |]
  |]

(* [image] is the original image. [width] and [height] are the model [width] on
   [height] relative to which bounding boxes have been computed.
*)
let report predictions ~image ~width ~height =
  Tensor.print_shape ~name:"predictions" predictions;
  let bboxes =
    List.init
      (Tensor.shape2_exn predictions |> fst)
      ~f:(fun index ->
        let predictions = Tensor.get predictions index |> Tensor.to_float1_exn in
        let confidence = predictions.(4) in
        if Float.( > ) confidence confidence_threshold
        then (
          let xmin = predictions.(0) -. (predictions.(2) /. 2.) in
          let ymin = predictions.(1) -. (predictions.(3) /. 2.) in
          let xmax = predictions.(0) +. (predictions.(2) /. 2.) in
          let ymax = predictions.(1) +. (predictions.(3) /. 2.) in
          let best_class_index =
            Array.foldi predictions ~init:5 ~f:(fun index max_index v ->
                if index > 5 && Float.( < ) predictions.(max_index) v
                then index
                else max_index)
          in
          let class_confidence = predictions.(best_class_index) in
          let class_index = best_class_index - 5 in
          if Float.( > ) class_confidence 0.
          then Some { confidence; xmin; ymin; xmax; ymax; class_index; class_confidence }
          else None)
        else None)
    |> List.filter_opt
  in
  let bboxes =
    (* Group bboxes by class-index. *)
    List.map bboxes ~f:(fun bbox -> bbox.class_index, (bbox.confidence, bbox))
    |> Map.of_alist_multi (module Int)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (_, bboxes) ->
           (* NMS suppression. *)
           let bboxes = List.sort bboxes ~compare:Caml.compare |> List.rev_map ~f:snd in
           List.fold bboxes ~init:[] ~f:(fun acc_bboxes bbox ->
               let drop =
                 List.exists acc_bboxes ~f:(fun b ->
                     Float.( > ) (iou b bbox) nms_threshold)
               in
               if drop then acc_bboxes else bbox :: acc_bboxes))
  in
  let image = Tensor.(to_type image ~type_:Float / f 255.) in
  let _, _, initial_height, initial_width = Tensor.shape4_exn image in
  let resize_and_clamp v ~initial_max ~max =
    Int.of_float (v *. Float.of_int initial_max /. Float.of_int max)
    |> Int.max 0
    |> Int.min (initial_max - 1)
  in
  List.iter bboxes ~f:(fun b ->
      let xmin = resize_and_clamp b.xmin ~initial_max:initial_width ~max:width in
      let xmax = resize_and_clamp b.xmax ~initial_max:initial_width ~max:width in
      let ymin = resize_and_clamp b.ymin ~initial_max:initial_height ~max:height in
      let ymax = resize_and_clamp b.ymax ~initial_max:initial_height ~max:height in
      let color = colors.(b.class_index % Array.length colors) in
      let color = Tensor.(of_float1 color |> reshape ~shape:[ 1; 3; 1; 1 ]) in
      let draw_rect xmin xmax ymin ymax =
        Tensor.narrow image ~dim:3 ~start:xmin ~length:(xmax - xmin)
        |> Tensor.narrow ~dim:2 ~start:ymin ~length:(ymax - ymin)
        |> Tensor.copy_ ~src:color
      in
      draw_rect xmin xmax ymin (Int.min (ymin + 2) ymax);
      draw_rect xmin xmax (Int.max ymin (ymax - 2)) ymax;
      draw_rect (Int.max xmin (xmax - 2)) xmax ymin ymax;
      draw_rect xmin (Int.min (xmin + 2) xmax) ymin ymax;
      Stdio.printf
        "%s %.2f %.2f (%d %d %d %d)\n%!"
        classes.(b.class_index)
        b.confidence
        b.class_confidence
        xmin
        xmax
        ymin
        ymax);
  Image.write_image Tensor.(image * f 255.) ~filename:"output.jpg"

let () =
  if Array.length Sys.argv <> 3
  then Printf.failwithf "usage: %s yolo-v3.ot input.png" Sys.argv.(0) ();
  (* Build the model. *)
  let vs = Var_store.create ~name:"rn" ~device:Cpu () in
  let darknet = Darknet.parse_config config_filename in
  Stdio.printf "%d blocks in %s\n%!" (List.length darknet.blocks) config_filename;
  let model = Darknet.build_model vs darknet in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  (* Load the image. *)
  let width, height = Darknet.width darknet, Darknet.height darknet in
  let image = Image.load_image Sys.argv.(2) |> Or_error.ok_exn in
  let resized_image = Image.resize image ~width ~height in
  let resized_image = Tensor.(to_type resized_image ~type_:Float / f 255.) in
  (* Apply the model. *)
  let predictions = Layer.apply_ model resized_image ~is_training:false in
  Tensor.squeeze predictions |> report ~image ~width ~height
