from random import randint
from functools import reduce

d = 4#randint(3, 6)

n = [randint(2, 3) for _ in range(d)]

print(d)
print(*n)

m = reduce(lambda a, b: a * b, n)

print(*[randint(0, 100) for _ in range(m)])