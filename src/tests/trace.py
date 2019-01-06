import torch
def foo(x, y):
  return 2 * x + y
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
traced_foo.save("foo.pt")

def foo2(x, y):
  return (2 * x + y, x - y)
traced_foo2 = torch.jit.trace(foo2, (torch.rand(3), torch.rand(3)))
traced_foo2.save("foo2.pt")

def foo3_fn(x):
  result = x[0]
  for i in range(x.size(0)):
    if i != 0:
      result = result * x[i]
  return result

foo3 = torch.jit.script(foo3_fn)

foo3.save("foo3.pt")
