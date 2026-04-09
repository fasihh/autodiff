from fygrad.node import Node

x = Node("x", 2)
y = Node("y", 2)


z = x**2 + Node.exp(y**2)
print(z)

z.backward()

print("dx:", x.grad)
print("dy:", y.grad)
