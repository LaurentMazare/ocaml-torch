ALL = examples/basics/torch_basics.exe \
      examples/mnist/linear.exe \
      examples/mnist/nn.exe \
      examples/mnist/conv.exe \
      examples/gan/mnist_gan.exe \
      examples/gan/mnist_cgan.exe \
      examples/gan/mnist_dcgan.exe \
      examples/cifar/conv.exe \
      examples/cifar/resnet.exe \
      examples/cifar/densenet.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

gen: .FORCE
	dune build src/gen/gen.exe
	./_build/default/src/gen/gen.exe

utop: .FORCE
	dune build @install
	dune build bin/utop_torch.bc
	dune exec bin/utop_torch.bc

jupyter: .FORCE
	dune build @install
	dune exec jupyter lab

clean:
	rm -Rf _build/ *.exe

.FORCE:
