# ocaml-torch
Experimental torch bindings in ocaml, the libtorch library can be downloaded from the [PyTorch website](https://pytorch.org/resources) ([latest cpu version](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)).

Extract the libtorch library in a `pytorch directory` then to build run:

```bash
LD_LIBRARY_PATH=libtorch/lib:$LD_LIBRARY_PATH LIBRARY_PATH=libtorch/lib:$LIBRARY_PATH CPATH=libtorch/include:$CPATH make
```

Then the example can be run with:
```bash
LD_LIBRARY_PATH=$HOME/tmp/libtorch/lib:$LD_LIBRARY_PATH ./_build/default/examples/basics/torch_tensor.exe

```
