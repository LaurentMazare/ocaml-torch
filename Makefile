ALL = examples/basics/torch_basics.exe \
      examples/mnist/linear.exe \
      examples/mnist/nn.exe \
      examples/mnist/conv.exe \
      examples/gan/mnist_gan.exe \
      examples/gan/mnist_cgan.exe \
      examples/gan/mnist_dcgan.exe \
      examples/gan/progressive_growing_gan.exe \
      examples/gan/relativistic_dcgan.exe \
      examples/gan/gan_stability.exe \
      examples/cifar/cifar_train.exe \
      examples/char_rnn/char_rnn.exe \
      examples/neural_transfer/neural_transfer.exe \
      examples/pretrained/finetuning.exe \
      examples/pretrained/predict.exe \
      bin/tensor_tools.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

gen: .FORCE
	dune build src/gen/gen.exe
	./_build/default/src/gen/gen.exe

utop: .FORCE
	dune utop

jupyter: .FORCE
	dune build @install
	dune exec jupyter lab

clean:
	rm -Rf _build/ *.exe

.FORCE:
