import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE

def compute_embeddings(dataloader, model):
    model.eval()
    embeddings = np.empty((1,5))
    labels = np.empty((1,))

    for i, (keypoints, targets) in enumerate(dataloader):
        input = torch.autograd.Variable(keypoints)
        targets = torch.autograd.Variable(targets)
        with torch.no_grad():
            output = model(input)

        embedding = output.numpy()
        embeddings = np.concatenate((embeddings, embedding))
        targets = targets.numpy()
        labels = np.concatenate((labels, targets))
    return embeddings, labels

# Utility function to visualize the outputs of PCA and t-SNE
def tsne_plot(x, labels):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = ["F", "G", "L", "O", "Y"]

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        # txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt = ax.text(xtext, ytext, txts[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def run_tsne_on_embeddings(embeddings):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(embeddings)
    pass



if __name__ == '__main__':
    from datasets.hgm import HgmDataset
    from models.KeypointToGestureStatic import KeypointToGestureStatic
    import numpy as np

    # Load a dataset of keypoints
    dataset = HgmDataset()
    total_size = len(dataset)
    print(f'Total dataset size {total_size}')
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             pin_memory=True)
    #Load model
    model = KeypointToGestureStatic()
    model.load_model_for_inference('/home/thwang/dev/insideiq/HandGestureRecognizer/checkpoints/k2gs_model_only_best.pt')

    #Compute embeddings
    embeddings, labels = compute_embeddings(dataloader, model)
    print(embeddings.shape)
    print(labels.shape)




