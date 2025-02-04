import matplotlib.pyplot as plt

billions_of_tokens = [1, 2, 4, 6, 8]
full_pt_lambada = [106.18, 105.56, 101.28, 85.73, 83.58]
relora_lambada = [100.49, 95.53, 93.20, 96.02, 91.76]

# Plot full_pt_lambada and relora_lambada against billions_of_tokens on the same plot

plt.plot(billions_of_tokens, full_pt_lambada, label='Full PT Lambada')
plt.plot(billions_of_tokens, relora_lambada, label='Relora Lambada')
plt.xlabel('Billions of tokens trained on post growth of checkpoint at 300-x tokens')
plt.ylabel('Perplexity on lambada_openai')
plt.legend()

# Save the plot
plt.savefig('plots/lambada_relora_full.png', dpi=300)