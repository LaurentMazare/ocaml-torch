ALL = examples/basics/torch_tensor.exe examples/mnist/linear.exe

%.exe: .FORCE
	dune build $@

all: .FORCE
	dune build $(ALL)

clean:
	rm -Rf _build/ *.exe

.FORCE:
