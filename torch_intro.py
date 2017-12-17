import torch

print("http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py")

print("Construct a 5x3 matrix, uninitialized:")
x = torch.Tensor(5, 3)
print(x)

print("Construct a randomly initialized matrix")
x = torch.rand(5, 3)
print(x)

print("Addition: syntax 1")
y = torch.rand(5, 3)
print(x + y)

print("Addition: syntax 2")
print(torch.add(x, y))

print("Addition: in-place")
# adds x to y
y.add_(x)
print(y)


from torch.autograd import Variable
print("http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py")

print("CREATING A variable")
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

print("You can do many crazy things with autograd!")
x = torch.ones(3)
x = Variable(x, requires_grad=True)
y = x * 2
i = 0
print("before:")
print(x)
while y.data.norm() < 1000:
    y = y * 2
    i += 1
print(i)
print(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
print(x.grad)
