ALL = examples/basics/torch_basics.exe \
      examples/mnist/linear.exe \
      examples/mnist/nn.exe \
      examples/mnist/conv.exe \
      examples/gan/mnist_gan.exe \
      examples/gan/mnist_dcgan.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

gen: .FORCE
	dune build src/gen/gen.exe
	./_build/default/src/gen/gen.exe

clean:
	rm -Rf _build/ *.exe

.FORCE:
