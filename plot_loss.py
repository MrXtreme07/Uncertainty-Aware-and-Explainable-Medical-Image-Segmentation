import matplotlib.pyplot as plt

# Read losses
losses = []
with open("loss.txt", "r") as f:
    for line in f:
        losses.append(float(line.strip()))

# Plot
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()

plt.show()