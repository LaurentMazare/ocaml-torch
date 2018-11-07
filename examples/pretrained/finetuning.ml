(* Finetuning an ImageNet trained model.
   This is similar to the PyTorch tutorial "Finetuning Torchvision Models"
     https://download.pytorch.org/tutorial/hymenoptera_data.zip

   The dataset can be found here:
     https://download.pytorch.org/tutorial/hymenoptera_data.zip

   Network weights can be found here:
     https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
*)

open Base
open Torch

let load_dataset path =
  Dataset_helper.read_with_cache
    ~cache_file:(Printf.sprintf "%s/cache.ot" path)
    ~read:(fun () ->
      let load s =
        Imagenet.load_images ~dir:(Printf.sprintf "%s/%s" path s)
      in
      let train0 = load "train/ants" in
      let train1 = load "train/bees" in
      let val0 = load "val/ants" in
      let val1 = load "val/bees" in
      let labels img0 img1 =
        let n0 = Tensor.shape img0 |> List.hd_exn in
        let n1 = Tensor.shape img1 |> List.hd_exn in
        Tensor.cat ~dim:0
          [ Tensor.zeros [ n0 ] ~kind:Int64
          ; Tensor.ones [ n1 ] ~kind:Int64
          ]
      in
      { Dataset_helper.train_images = Tensor.cat [ train0; train1 ] ~dim:0
      ; train_labels = labels train0 train1
      ; test_images = Tensor.cat [ val0; val1 ] ~dim:0
      ; test_labels = labels val0 val1
      })

let () =
  if Array.length Sys.argv <> 3
  then Printf.sprintf "usage: %s resnet18.ot dataset-path" Sys.argv.(0) |> failwith;
  let dataset = load_dataset Sys.argv.(2) in
  Tensor.print_shape ~name:"train-images" dataset.train_images;
  Tensor.print_shape ~name:"test-images" dataset.test_images;
  let frozen_vs = Var_store.create ~name:"rn" () in
  let train_vs = Var_store.create ~name:"rn-vs" () in
  let model = Resnet.resnet18 frozen_vs in
  let fc = Layer.linear train_vs ~input_dim:52 2 in
  let model xs = Layer.apply_ model xs ~is_training:false |> Layer.apply fc in
  Stdio.printf "Loading weights from %s\n%!" Sys.argv.(1);
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars frozen_vs) ~filename:Sys.argv.(1);
  ignore (model, ())

