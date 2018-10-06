open Base
open Stdio

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

let rec contains_string ~str = function
  | `A l -> List.exists l ~f:(contains_string ~str)
  | `O l -> List.exists l ~f:(fun (_, y) -> contains_string y ~str)
  | `String s when String.(=) s str -> true
  | _ -> false

module Function = struct
  type arg =
    { arg_name : string
    ; arg_type : string
    ; default_value : string option
    }

  type t =
    { name : string
    ; args : arg list
    ; returns : string
    }
end

let run filename =
  let functions =
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
  printf "Read %s, got %d functions.\n%!" filename (List.length functions);
  let functions =
    List.filter_map functions ~f:(fun (func, variants) ->
      let has_function =
        match variants with
        | [] -> true
        | variants -> List.exists variants ~f:(contains_string ~str:"function")
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
          if String.is_substring args ~substring:"std::" || String.is_empty args
          then None
          else
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
                  | Some (arg_type, arg_name) -> Some { Function.arg_name; arg_type; default_value }
                  | None ->
                    printf "Unhandled argument format for %s: <%s>.\n%!" func_name arg;
                    None
              )
            in
            Some { Function.name = func_name; args; returns })
      else None
    )
  in
  printf "Generating code for %d functions.\n%!" (List.length functions)

let () = run "data/native_functions.yaml"
