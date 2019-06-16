(* The pre-trained weights can be downloaded here:
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot
*)
open Base
open Torch
open Torch_vision

let config_filename = "examples/yolo/yolo-v3.cfg"
let confidence_threshold = 0.5
let nms_threshold = 0.4
let classes = Coco_names.names

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
  let model = Darknet.build_model vs darknet in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:Sys.argv.(1);
  (* Load the image. *)
  let width, height = Darknet.width darknet, Darknet.height darknet in
  let image = Image.load_image Sys.argv.(2) |> Or_error.ok_exn in
  let resized_image = Image.resize image ~width ~height in
  let resized_image = Tensor.(to_type resized_image ~type_:Float / f 255.) in
  (* Apply the model. *)
  let predictions = Layer.forward_ model resized_image ~is_training:false in
  Tensor.squeeze predictions |> report ~image ~width ~height
