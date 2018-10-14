(* TODO: add the possibility to only keep a fixed number of checkpoints. *)
open Base

let latest_index_and_filename ~checkpoint_base =
  let dirname = Caml.Filename.dirname checkpoint_base in
  let basename = Caml.Filename.basename checkpoint_base in
  Caml.Sys.readdir dirname
  |> Array.to_list
  |> List.filter_map ~f:(fun filename ->
      match String.chop_prefix filename ~prefix:(basename ^ ".") with
      | None -> None
      | Some suffix ->
        try
          Some (Int.of_string suffix, Caml.Filename.concat dirname filename)
        with _ -> None)
  |> List.sort ~compare:Caml.Pervasives.compare
  |> List.last

let loop
      ~start_index
      ~end_index
      ~named_tensors
      ~checkpoint_base
      ?(checkpoint_every = `seconds 600.)
      f
  =
  if start_index < 0
  then raise (Invalid_argument (Printf.sprintf "negative start_index %d" start_index));
  let temp_checkpoint = checkpoint_base ^ ".tmp" in
  let latest_index_and_filename = latest_index_and_filename ~checkpoint_base in
  Option.iter latest_index_and_filename ~f:(fun (latest_index, filename) ->
    Stdio.eprintf "Restoring checkpoint for index %d from '%s'.\n%!" latest_index filename;
    Serialize.load_multi_ ~named_tensors ~filename);

  let start_index =
    Option.value_map latest_index_and_filename ~default:start_index
      ~f:(fun (index, _) -> index + 1)
  in
  let save ~suffix =
    Serialize.save_multi ~named_tensors ~filename:temp_checkpoint;
    Unix.rename
      temp_checkpoint
      (Printf.sprintf "%s.%s" checkpoint_base suffix)
  in
  let last_checkpoint_time = ref (Unix.time ()) in
  for index = start_index to end_index do
    f ~index;
    let should_checkpoint =
      match checkpoint_every with
      | `seconds seconds -> Float.(>) (Unix.time () -. !last_checkpoint_time) seconds
      | `iters iters -> index % iters = 0
    in
    if should_checkpoint
    then begin
      save ~suffix:(Int.to_string index);
      last_checkpoint_time := Unix.time ()
    end
  done;
  save ~suffix:"final"

