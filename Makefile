ALL = examples/char_rnn/char_rnn.exe \
      examples/cifar/cifar_train.exe \
      examples/gan/began.exe \
      examples/gan/gan_stability.exe \
      examples/gan/mnist_cgan.exe \
      examples/gan/mnist_dcgan.exe \
      examples/gan/mnist_gan.exe \
      examples/gan/progressive_growing_gan.exe \
      examples/gan/relativistic_dcgan.exe \
      examples/jit/load_and_run.exe \
      examples/mnist/conv.exe \
      examples/mnist/linear.exe \
      examples/mnist/nn.exe \
      examples/neural_transfer/neural_transfer.exe \
      examples/pretrained/finetuning.exe \
      examples/pretrained/predict.exe \
      examples/yolo/yolo.exe \
      examples/vae/vae.exe \
      examples/translation/seq2seq.exe \
      examples/transformer/transformer.exe \
      bin/tensor_tools.exe

RL = examples/reinforcement-learning/dqn.exe \
     examples/reinforcement-learning/dqn_atari.exe \
     examples/reinforcement-learning/dqn_pong.exe \
     examples/reinforcement-learning/policy_gradient.exe \
     examples/reinforcement-learning/ppo.exe \
     examples/reinforcement-learning/a2c.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

rl: .FORCE
	dune build $(RL)

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
