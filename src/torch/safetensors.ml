open! Base

exception Read_error of string

let read_error fmt = Printf.ksprintf (fun s -> raise (Read_error s)) fmt

let map_file file_descr ~pos ~len =
  Unix.map_file
    file_descr
    ~pos:(Int_conversions.int_to_int64 pos)
    Bigarray.Int8_unsigned
    C_layout
    false
    [| len |]

let read ?only filename =
  let only = Option.map only ~f:(Hash_set.of_list (module String)) in
  Stdio.In_channel.with_file filename ~binary:true ~f:(fun in_c ->
    let header_size =
      match Caml.really_input_string in_c 8 with
      | header_size -> header_size
      | exception _ -> read_error "unexpected eof while reading header size"
    in
    let header_size =
      Int_repr.String.get_uint64_le header_size ~pos:0
      |> Int_repr.Uint64.to_base_int64_exn
      |> Base.Int_conversions.int64_to_int_exn
    in
    let header =
      match Caml.really_input_string in_c header_size with
      | header -> header
      | exception _ -> read_error "unexpected eof while reading header len:%d" header_size
    in
    let header =
      match Yojson.Safe.from_string header with
      | `Assoc assoc -> assoc
      | _ -> read_error "header is not a json object"
    in
    let fd = Unix.descr_of_in_channel in_c in
    List.filter_map header ~f:(function
      | "__metadata__", _ -> None
      | tensor_name, `Assoc details
        when Option.value_map only ~default:true ~f:(fun only ->
               Hash_set.mem only tensor_name) ->
        let details = Hashtbl.of_alist_exn (module String) details in
        let packed_ty =
          match Hashtbl.find details "dtype" with
          | None -> read_error "missing dtype for %s" tensor_name
          | Some (`String "F16") -> Torch_core.Kind.T Half
          | Some (`String "F32") -> T Float
          | Some (`String "F64") -> T Double
          | Some (`String "I64") -> T Int64
          | Some (`String "I32") -> T Int
          | Some (`String "I16") -> T Int16
          | Some (`String "I8") -> T Int8
          | Some (`String "U8") -> T Uint8
          | Some dtype ->
            read_error
              "unexpected dtype for %s: %s"
              tensor_name
              (Yojson.Safe.to_string dtype)
        in
        let start_offset, stop_offset =
          match Hashtbl.find details "data_offsets" with
          | None -> read_error "missing data_offsets for %s" tensor_name
          | Some (`List [ `Int start; `Int stop ]) -> start, stop
          | Some dtype ->
            read_error
              "unexpected data_offsets for %s: %s"
              tensor_name
              (Yojson.Safe.to_string dtype)
        in
        let shape =
          match Hashtbl.find details "shape" with
          | None -> read_error "missing shape for %s" tensor_name
          | Some (`List dims) ->
            List.map dims ~f:(function
              | `Int i -> i
              | other ->
                read_error
                  "unexpected shape for %s: %s"
                  tensor_name
                  (Yojson.Safe.to_string other))
          | Some dtype ->
            read_error
              "unexpected shape for %s: %s"
              tensor_name
              (Yojson.Safe.to_string dtype)
        in
        let src =
          map_file
            fd
            ~pos:(8 + header_size + start_offset)
            ~len:(stop_offset - start_offset)
        in
        let tensor = Tensor.of_bigarray_bytes src packed_ty ~shape in
        Some (tensor_name, tensor)
      | _, `Assoc _ -> None
      | tensor_name, _ ->
        read_error "header details for %s is not a json object" tensor_name))
