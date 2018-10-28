let parse_and_execute str =
  Lexing.from_string str
  |> !Toploop.parse_toplevel_phrase
  |> Toploop.execute_phrase true Format.err_formatter
  |> fun (_:bool) -> ()

let register_pp str =
  Printf.sprintf "#install_printer %s;;" str |> parse_and_execute

let register_all_pps () =
  register_pp "Torch.Tensor.pp"

let toplevel_start () =
  Toploop.initialize_toplevel_env ();
  parse_and_execute "#use \"topfind\";;";
  parse_and_execute "#require \"torch\";;";
  register_all_pps ()
