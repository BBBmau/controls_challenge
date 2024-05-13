import numpy as np
rews = np.array([1,2,3,4,5,6,7,8])
discounts = np.array([1,2,3,4,5,6,7,8])

Rn = []
for i in range(len(rews)):
    print(rews[i:] * discounts[:len(rews)-i])
Rn += [np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]
print(Rn)