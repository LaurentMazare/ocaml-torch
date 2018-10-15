ALL = examples/basics/torch_basics.exe \
      examples/mnist/linear.exe \
      examples/mnist/nn.exe \
      examples/mnist/conv.exe \
      examples/gan/mnist_gan.exe \
      examples/gan/mnist_cgan.exe \
      examples/gan/mnist_dcgan.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

gen: .FORCE
	dune build src/gen/gen.exe
	./_build/default/src/gen/gen.exe

utop: .FORCE
	dune build bin/utop_torch.bc
	CAML_LD_LIBRARY_PATH=_build/default/src/wrapper:$(CAML_LD_LIBRARY_PATH) dune exec bin/utop_torch.bc

clean:
	rm -Rf _build/ *.exe

.FORCE:
