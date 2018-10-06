open Base
open Stdio

let unsupported_functions =
  Set.of_list (module String) [ "bincount"; "stft"; "group_norm"; "layer_norm"; "rot90"; "t" ]

let yaml_error yaml ~msg =
  Printf.sprintf "%s, %s" msg (Yaml.to_string_exn yaml)
  |> failwith

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_multi (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  | yaml -> yaml_error yaml ~msg:"expected string"

let contains_substring yaml ~substring =
  let rec walk = function
  | `A l -> List.exists l ~f:walk
  | `O l -> List.exists l ~f:(fun (_, y) -> walk y)
  | `String s when String.is_substring s ~substring -> true
  | _ -> false
  in
  walk yaml

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Double
    | Tensor
    | IntList
    | TensorOptions

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; default_value : string option
    }

  type t =
    { name : string
    ; args : arg list
    ; returns : string
    }

  let arg_type_of_string str =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some Int64
    | "double" -> Some Double
    | "tensor" -> Some Tensor
    | "tensoroptions" -> Some TensorOptions
    | _ ->
      if String.is_prefix str ~prefix:"IntList"
      then Some IntList
      else None

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | IntList ->
        Printf.sprintf "int *%s_data, int %s_len" arg_name arg_name
      | otherwise ->
        let simple_type_cstring =
          match otherwise with
          | Bool -> "int"
          | Int64 -> "int64_t"
          | Double -> "double"
          | Tensor -> "tensor"
          | TensorOptions -> "int" (* only Kind for now. *)
          | IntList -> assert false
        in
        Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
      match arg_type with
      | Tensor -> "*" ^ arg_name
      | Bool -> "(bool)" ^ arg_name
      | IntList -> Printf.sprintf "of_carray(%s_data, %s_len)" arg_name arg_name
      | TensorOptions -> Printf.sprintf "torch::ScalarType(%s)" arg_name
      | _ -> arg_name)
    |> String.concat ~sep:", "

  let stubs_signature t =
    List.concat_map t.args ~f:(fun arg ->
      match arg.arg_type with
      | Bool -> ["int"]
      | Int64 -> ["int64_t"]
      | Double -> ["double"]
      | Tensor -> ["t"]
      | TensorOptions -> ["int"]
      | IntList -> ["ptr int"; "int"]
    )
    |> String.concat ~sep:" @-> "
    |> Printf.sprintf "%s @-> returning t"

  let replace_map =
    Map.of_alist_exn (module String)
      [ "end", "end_"
      ]

  let caml_name arg =
    Map.find replace_map arg.arg_name |> Option.value ~default:arg.arg_name
    |> String.lowercase

  let caml_args t =
    List.map t.args ~f:caml_name
    |> String.concat ~sep:" "

  let caml_binding_args t =
    List.map t.args ~f:(fun arg ->
      let name = caml_name arg in
      match arg.arg_type with
      | IntList ->
        Printf.sprintf
          "(CArray.of_list int %s |> CArray.start) (List.length %s)"
          name name
      | Bool -> Printf.sprintf "(if %s then 1 else 0)" name
      | TensorOptions -> Printf.sprintf "(Kind.to_int %s)" name
      | _ -> name)
    |> String.concat ~sep:" "
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    In_channel.with_file filename ~f:In_channel.input_all
    |> Yaml.of_string_exn
    |> extract_list
    |> List.map ~f:(fun yaml ->
      let map = extract_map yaml in
      let func =
        match Map.find_exn map "func" with
        | [] -> assert false
        | [func] -> extract_string func
        | _ :: _ :: _ -> yaml_error yaml ~msg:"multiple func"
      in
      func, Map.find map "variants" |> Option.value ~default:[])
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun (func, variants) ->
    let has_function =
      match variants with
      | [] -> true
      | variants -> List.exists variants ~f:(contains_substring ~substring:"function")
    in
    if has_function
    then
      Option.bind (String.substr_index func ~pattern:"->") ~f:(fun arrow_index ->
        let lhs = String.prefix func arrow_index |> String.strip in
        let returns = String.drop_prefix func (arrow_index + 2) |> String.strip in
        let func_name, args = String.lsplit2_exn lhs ~on:'(' in
        assert (Char.(=) args.[String.length args - 1] ')');
        let args = String.drop_suffix args 1 in
        (* Remove args that contain a std::array<> because of the extra commas... *)
        if String.is_substring args ~substring:"std::"
        || String.is_empty args
        || String.(<>) returns "Tensor"
        || Char.(=) func_name.[0] '_'
        || Set.mem unsupported_functions func_name
        then None
        else
          try
            let args =
              String.split args ~on:','
              |> List.filter_map ~f:(fun arg ->
                let arg = String.strip arg in
                if String.(=) arg "*"
                then None
                else
                  let arg, default_value =
                    match String.split arg ~on:'=' with
                    | [arg] -> String.strip arg, None
                    | [arg; default_value] -> String.strip arg, Some (String.strip default_value)
                    | _ -> Printf.sprintf "unexpected arg format %s" arg |> failwith
                  in
                  match String.rsplit2 arg ~on:' ' with
                  | None ->
                    Printf.sprintf "Unhandled argument format for %s: <%s>.\n%!" func_name arg
                    |> failwith
                  | Some (arg_type, arg_name) ->
                    match Func.arg_type_of_string arg_type with
                    | Some arg_type ->
                      Some { Func.arg_name; arg_type; default_value }
                    | None ->
                      if Option.is_some default_value
                      then None
                      else raise Not_a_simple_arg
              )
            in
            Some { Func.name = func_name; args; returns }
          with
          | Not_a_simple_arg -> None)
    else None
  )

let p out_channel s =
  Printf.ksprintf (fun line ->
    Out_channel.output_string out_channel line;
    Out_channel.output_char out_channel '\n') s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp.h") ~f:(fun out_cpp ->
    Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
      let pc s = p out_cpp s in
      let ph s = p out_h s in
      pc "";
      pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
      pc "";
      ph "";
      ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
      ph "";
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
        let { Func.name; _ } = func in
        let c_typed_args_list = Func.c_typed_args_list func in
        let c_args_list = Func.c_args_list func in
        pc "tensor atg_%s(%s) {" exported_name c_typed_args_list;
        pc "  PROTECT(";
        pc "    return new torch::Tensor(torch::%s(%s));" name c_args_list;
        pc "  )";
        pc "}";
        pc "";
        ph "tensor atg_%s(%s);" exported_name c_typed_args_list;
      )
    )
  )

let write_stubs funcs filename =
  Out_channel.with_file filename ~f:(fun out_channel ->
    let p s = p out_channel s in
    p "open Ctypes";
    p "";
    p "module C(F: Cstubs.FOREIGN) = struct";
    p "  open F";
    p "  type t = unit ptr";
    p "  let t : t typ = ptr void";
    Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
      p "  let %s =" exported_name;
      p "    foreign \"atg_%s\"" exported_name;
      p "    (%s)" (Func.stubs_signature func);
      p "";
    );
    p "end")

let write_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_channel ->
    let p s = p out_channel s in
    p "open Ctypes";
    p "";
    p "module C = Torch_bindings.C(Torch_generated)";
    p "open C.TensorG";
    p "";
    Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
      p "let %s %s =" exported_name (Func.caml_args func);
      p "  let t = %s %s in" exported_name (Func.caml_binding_args func);
      p "  Gc.finalise C.Tensor.free t;";
      p "  t";
      p "";
    )
  )

let run ~yaml_filename ~cpp_filename ~stubs_filename ~wrapper_filename =
  let funcs = read_yaml yaml_filename in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> String.lowercase func.name, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
      match funcs with
      | [] -> assert false
      | [ func ] -> [ name, func ]
      | funcs ->
        List.mapi funcs ~f:(fun i func ->
          Printf.sprintf "%s%d" name (i+1), func)
      )
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename;
  write_stubs funcs stubs_filename;
  write_wrapper funcs wrapper_filename

let () =
  run
    ~yaml_filename:"data/native_functions.yaml"
    ~cpp_filename:"src/wrapper/torch_api_generated"
    ~stubs_filename:"src/stubs/torch_bindings_generated.ml"
    ~wrapper_filename:"src/wrapper/wrapper_generated.ml"
