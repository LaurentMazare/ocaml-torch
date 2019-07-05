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
           (try Some (Int.of_string suffix, Caml.Filename.concat dirname filename) with
           | _ -> None))
  |> List.sort ~compare:Stdlib.compare
  |> List.last

let loop
    ~start_index
    ~end_index
    ~var_stores
    ~checkpoint_base
    ?only_keep
    ?(checkpoint_every = `seconds 600.)
    f
  =
  if start_index < 0 then Printf.invalid_argf "negative start_index %d" start_index ();
  Option.iter only_keep ~f:(fun only_keep ->
      if only_keep <= 0 then Printf.invalid_argf "non-positive only_keep %d" only_keep ());
  let temp_checkpoint = checkpoint_base ^ ".tmp" in
  let latest_index_and_filename = latest_index_and_filename ~checkpoint_base in
  let named_tensors =
    match var_stores with
    | [ vs ] -> Var_store.all_vars vs
    | var_stores ->
      List.concat_map var_stores ~f:(fun vs ->
          let vs_name = Var_store.name vs in
          Var_store.all_vars vs
          |> List.map ~f:(fun (name, tensor) ->
                 Printf.sprintf "%s:%s" vs_name name, tensor))
  in
  Option.iter latest_index_and_filename ~f:(fun (latest_index, filename) ->
      Stdio.eprintf
        "Restoring checkpoint for index %d from '%s'.\n%!"
        latest_index
        filename;
      Serialize.load_multi_ ~named_tensors ~filename);
  let start_index =
    Option.value_map latest_index_and_filename ~default:start_index ~f:(fun (index, _) ->
        index + 1)
  in
  let only_keep =
    Option.map only_keep ~f:(fun only_keep -> only_keep, Linked_queue.create ())
  in
  let save ~suffix =
    Serialize.save_multi ~named_tensors ~filename:temp_checkpoint;
    Unix.rename temp_checkpoint (Printf.sprintf "%s.%s" checkpoint_base suffix)
  in
  let save_index index =
    save ~suffix:(Int.to_string index);
    Option.iter only_keep ~f:(fun (only_keep, index_queue) ->
        Linked_queue.enqueue index_queue index;
        if Linked_queue.length index_queue > only_keep
        then
          Linked_queue.dequeue_exn index_queue
          |> Int.to_string
          |> Printf.sprintf "%s.%s" checkpoint_base
          |> Unix.unlink)
  in
  let last_checkpoint_time = ref (Unix.time ()) in
  for index = start_index to end_index do
    f ~index;
    let should_checkpoint =
      match checkpoint_every with
      | `seconds seconds -> Float.( > ) (Unix.time () -. !last_checkpoint_time) seconds
      | `iters iters -> index % iters = 0
    in
    if should_checkpoint
    then (
      save_index index;
      last_checkpoint_time := Unix.time ())
  done;
  save ~suffix:"final"
