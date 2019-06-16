(* Translation with a Sequence to Sequence Model and Attention.

   This follows the line of the PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

   The dataset can be downloaded from the following link:
   https://download.pytorch.org/tutorial/data.zip
   The eng-fra.txt file should be moved in the data directory.
*)
open! Base
open Torch

let () =
  let _device = Device.cuda_if_available () in
  let dataset = Dataset.create ~input_lang:"eng" ~output_lang:"fra" |> Dataset.reverse in
  let input_lang = Dataset.input_lang dataset in
  let output_lang = Dataset.output_lang dataset in
  Stdio.printf "Input:  %s %d words.\n%!" (Lang.name input_lang) (Lang.length input_lang);
  Stdio.printf
    "Output: %s %d words.\n%!"
    (Lang.name output_lang)
    (Lang.length output_lang);
  ()
