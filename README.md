# ocaml-torch
Experimental [PyTorch](https://pytorch.org) bindings in ocaml using [C++ API](https://pytorch.org/cppdocs/).
The libtorch library can be downloaded from the [PyTorch website](https://pytorch.org/resources) ([latest cpu version](https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)).

## Usage
Extract the libtorch library in a `pytorch directory` then to build examples run:

```bash
LD_LIBRARY_PATH=libtorch/lib:$LD_LIBRARY_PATH \
LIBRARY_PATH=libtorch/lib:$LIBRARY_PATH \
CPATH=libtorch/include:libtorch/include/torch/csrc/api/include:$CPATH \
make all
```

After that examples can be run with:
```bash
LD_LIBRARY_PATH=$HOME/tmp/libtorch/lib:$LD_LIBRARY_PATH \
./_build/default/examples/basics/torch_tensor.exe
```

**TODO:** generate the bindings automatically from the [yaml function descriptions](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml) ?
