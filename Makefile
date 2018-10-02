ALL = examples/basics/torch_tensor.exe

%.exe: .FORCE
	dune build $@

clean:
	rm -Rf _build/ *.exe

.FORCE:

all: .FORCE
	dune build $(ALL)
