from sklearn import covariance
from sklearn import cluster
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import stock_market_structure as SmS

def main():
    print("\nSymbols\n", SmS.symbols)
    print("\nNames\n", SmS.names)
    print("\nVariation (closed - open prices) shape:", SmS.variation.shape)
    print(SmS.variation)
    #print(SmS.closed_prices.shape)

    # Fit a covariance model
    alphas = np.logspace(-1.5, 1, num=10)
    edge_model = covariance.GraphicalLassoCV(alphas=alphas)
    X = SmS.variation.copy().T
    X/= X.std(axis=0)
    edge_model.fit(X)
    print("\nEdge model fitted.")
    print("Covariance matrix shape:", edge_model.covariance_.shape)

    _, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
    n_labels = labels.max() 
    print(f"\nFound {n_labels + 1} clusters in the stock market structure.")
    for i in range(n_labels + 1):
        print(f"Label {i+1}: {np.sum(labels == i)} stocks")
        print(f"Cluster {i+1}: {', '.join(SmS.names[labels == i])}")
        print("===")
    
    #Time for visualisation!
    print("\nNow start visualizing the stock market structure...")
    # Create a 2D embedding of the stocks/nodes using Locally Linear Embedding
    # This will help us visualize the graph structure in 2D
    node_pos = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver="dense", n_neighbors=6
    )
    embedding = node_pos.fit_transform(X.T).T

    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax=plt.axes([0.0, 0.0, 1.0, 1.0])
    plt.axis("off")

    # Plot the graph of partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations*=d
    partial_correlations*=d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

    #Plot the nodes using the embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d**2, c=labels, cmap=plt.cm.nipy_spectral)

    # Plot the edges
    start_indx, end_indx = non_zero.nonzero()
    segments = [
        [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_indx, end_indx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7*values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    #Add label to each node, avoiding overlaps!  
    for index, (name, label, (x,y)) in enumerate(zip(SmS.names, labels, embedding.T)):
        dx = x- embedding[0]
        dx[index] = 1
        dy= y - embedding[1]
        dy[index] = 1
        this_dx=dx[np.argmin(np.abs(dy))]
        this_dy=dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            ha = "left"
            x=x+0.002
        else:
            ha = "right"
            x=x-0.002
        if this_dy > 0:
            va = "bottom"
            y=y+0.002
        else:
            va = "top"
            y=y-0.002
        plt.text(x, y, name, ha=ha, va=va, size=10, bbox=dict(facecolor='w', 
                                                              edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                                                              alpha=0.6)
                                                              )

    plt.xlim(embedding[0].min() - 0.15*np.ptp(embedding[0]), embedding[0].max() + 0.10*np.ptp(embedding[0]))
    plt.ylim(embedding[1].min() - 0.03*np.ptp(embedding[1]), embedding[1].max() + 0.03*np.ptp(embedding[1]))

    plt.savefig("stock_market_structure.png", dpi=300)

    print("\nStock market structure visualized and saved as 'stock_market_structure.png'.")
    print("---done---")

if __name__ == "__main__":
    main()
