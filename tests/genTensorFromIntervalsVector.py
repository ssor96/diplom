from random import randint

n = 5#randint(4, 7)

borders = [sorted([randint(0, 100) / 100 + randint(-100, 100) for _ in range(2)]) for i in range(n)]

p = 20#randint(4, 9)

print(n + 1)

sizes = [n] + [p] * n

print(*sizes)

def get(a, b, i):
    return a + (b - a) * i / p


def pr(cur, l = []):
    if cur == n + 1:
        print(get(*borders[l[0]], l[l[0] + 1]), end=' ')
        return
    l.append(0)
    for i in range(sizes[cur]):
        l[-1] = i
        pr(cur + 1, l)
    del l[-1]


pr(0)
print()