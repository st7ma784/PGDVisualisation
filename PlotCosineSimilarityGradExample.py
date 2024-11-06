import numpy as np
import torch
import matplotlib.pyplot as plt

# Generate 64 random points

random_points = (np.random.rand(3, 2)*2) - 1 

# Define target point and prediction point
target_point = np.array([[0.5, 0.5]])
prediction_point = np.array([0.5, 0.55])

# Generate a ring of points around the prediction point
theta = np.linspace(0, 2 * np.pi, 100)
radius = 0.15
ring_points = np.array([ (radius * np.array([np.cos(t), np.sin(t)]))+prediction_point for t in theta])

# Calculate cross entropy loss for each point in the ring

normed_ring_points = ring_points / torch.norm(torch.tensor(ring_points), dim=-1, keepdim=True)
alltargets= torch.tensor(np.concatenate([target_point, random_points]))
normed_alltargets = alltargets / torch.norm(torch.tensor(alltargets), dim=-1, keepdim=True)
cosine_similarity = normed_ring_points @ normed_alltargets.T
losses = torch.nn.CrossEntropyLoss(reduction="none")(torch.tensor(cosine_similarity), torch.zeros(100,dtype=torch.long)).detach().numpy()

#plot losses
# plt.imshow(cosine_similarity, cmap='viridis', aspect='auto')
plt.plot(losses)
plt.xlabel('Index')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss for Ring Points')
plt.show()
plt.savefig('CrossEntropyLossRing.png')
#new plot
plt.figure()
# Plot the random points
plt.scatter(random_points[:, 0], random_points[:, 1], label='candidate Annotations', linewidths=0.1)



# Plot the prediction point
plt.scatter(prediction_point[0], prediction_point[1], color='Yellow', label='Prediction Point')
# Plot the target point
plt.scatter(target_point[:,0], target_point[:,1], color='Green',marker="x", label='Target Annotation')
# Plot the ring points with color based on cross entropy loss
norm = plt.Normalize(losses.min(), losses.max())
colors = plt.cm.magma(norm(losses))
# print(colors)


plt.scatter(ring_points[:, 0], ring_points[:, 1], c=colors, linewidths=0.2)

#show colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=norm)

plt.colorbar(sm)
# Add legend and show plot
plt.legend()
plt.xlabel('X-axis')
#axislimits
plt.xlim(-1,1)
plt.ylim(-1,1)

plt.ylabel('Y-axis')
plt.title('Cosine Similarity and Cross Entropy Loss Visualization')
plt.show()
# Save the plot
plt.savefig('CosineSimilarityGradExample.png')