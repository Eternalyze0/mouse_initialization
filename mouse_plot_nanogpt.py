import matplotlib.pyplot as plt

# MOUSE:
# step 2000: train loss 1.7245, val loss 1.8659
# step 2000: train loss 1.7661, val loss 1.9050
# step 2000: train loss 1.7641, val loss 1.8773

# BASELINE:
# step 2000: train loss 1.7867, val loss 1.9062
# step 2000: train loss 1.7609, val loss 1.9061
# step 2000: train loss 1.7685, val loss 1.8791

mouse = [1.8659, 1.9050, 1.8773]
baseline = [1.9062, 1.9061, 1.8791]

plt.plot(baseline, color='blue', label='baseline')
plt.plot(mouse, color='red', label='mouse')
plt.xlabel('Run Index')
plt.ylabel('Validation Loss')
plt.title('Baseline VS Mouse Initialization on nanoGPT')
plt.legend()
plt.savefig('nanogpt_mouse_plot.png')
