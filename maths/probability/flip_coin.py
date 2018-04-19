from maths.probability.comp_prob_inference import flip_fair_coin,flip_fair_coins,plot_discrete_histogram


print(flip_fair_coin())


flips = flip_fair_coins(100)
print(flips)

plot_discrete_histogram(flips)

plot_discrete_histogram(flips, frequency=True)

n = 100000
heads_so_far = 0
fraction_of_heads = []
for i in range(n):
    if flip_fair_coin() == 'heads':
        heads_so_far += 1
    fraction_of_heads.append(heads_so_far / (i+1))


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(range(1, n+1), fraction_of_heads)
plt.xlabel('Number of flips')
plt.ylabel('Fraction of heads')
plt.show()