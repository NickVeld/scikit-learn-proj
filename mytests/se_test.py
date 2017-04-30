#Made by Yaroslav, adopted by Nick
import dataset_generator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from time import time
from sklearn.manifold import SpectralEmbedding
Axes3D

#f = int(input())
f = 1000

def test(g=dataset_generator.Generator(), mode='show'):
    alpha_channel = 0.7
    spectre1 = np.array([np.array([1, 0, 0, alpha_channel]), np.array([1, 1, 0, alpha_channel])])
    spectre2 = np.array([np.array([0, 0, 1, alpha_channel]), np.array([0, 1, 0, alpha_channel])])
    spectre3 = np.array([np.array([1, 0, 1, alpha_channel]), np.array([0, 1, 1, alpha_channel])])
    n_neighbors = 10
    n_components = 2
    fig = plt.figure(figsize=(15, 8))
    se = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    '''--------------------------------------------------------------------------------------------------------------'''
    dataset, dataset_colors = g.generate_manifold(f, color_data=spectre1)
    ax = fig.add_subplot(241, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=dataset_colors, marker='s', edgecolors='none')

    dataset_embedding = se.fit_transform(dataset)
    ax = fig.add_subplot(245)
    ax.scatter(dataset_embedding[:, 0], dataset_embedding[:, 1], c=dataset_colors, cmap=plt.cm.Spectral, marker='s', edgecolors='none')
    '''--------------------------------------------------------------------------------------------------------------'''
    new_points, new_colors = g.generate_manifold(f, spectre2)

    new_points_embedding = np.array(list(
        se.transform(point.reshape(1, -1))[0]
        for point in new_points
    ))

    ax = fig.add_subplot(242, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=dataset_colors, marker='s', edgecolors='none')
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c=new_colors, marker='s', edgecolors='none')

    ax = fig.add_subplot(246)
    ax.scatter(dataset_embedding[:, 0], dataset_embedding[:, 1], c=dataset_colors, marker='s', edgecolor='none')
    ax.scatter(new_points_embedding[:, 0], new_points_embedding[:, 1], c=new_colors, marker='s', edgecolors='none')
    '''--------------------------------------------------------------------------------------------------------------'''
    re_colors = dataset_generator.Generator.generate_colors(None, dataset.shape[0], spectre3)
    re_embedding = np.array(list(
        se.transform(point.reshape(1, -1))[0]
        for point in dataset
    ))

    ax = fig.add_subplot(243, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=dataset_colors, marker='s', edgecolors='none')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=re_colors, marker='s', edgecolors='none')

    ax = fig.add_subplot(247)
    ax.scatter(dataset_embedding[:, 0], dataset_embedding[:, 1], c=dataset_colors, marker='s', edgecolor='none')
    ax.scatter(re_embedding[:, 0], re_embedding[:, 1], c=re_colors, marker='s', edgecolors='none')
    '''--------------------------------------------------------------------------------------------------------------'''

    dataset_reconstruction = np.array(list(
        se.inverse_transform(point.reshape(1, -1))[0]
        for point in dataset_embedding
    ))

    ax = fig.add_subplot(244)
    ax.scatter(dataset_embedding[:, 0], dataset_embedding[:, 1], c=dataset_colors, marker='s', edgecolor='none')


    ax = fig.add_subplot(248, projection='3d')
    ax.scatter(dataset_reconstruction[:, 0], dataset_reconstruction[:, 1], dataset_reconstruction[:, 2], c=dataset_colors, marker='s', edgecolors='none')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=re_colors, marker='s', edgecolors='none')
    '''--------------------------------------------------------------------------------------------------------------'''
    plt.axis('tight')

    if mode == 'show':
        plt.show()
        if f == 1000 or input('save? y/N') != 'y':
            pass
        else:
            save(fig, dataset, dataset_embedding, new_points, new_points_embedding)
    else:
        save(fig, dataset, dataset_embedding, new_points, new_points_embedding)


def save(fig, dataset, dataset_embedding, new_points, new_points_embedding):
    import os
    t = str(time())
    os.mkdir('output/' + t)
    fig.savefig('output/' + t + '/' + t + '.png')
    np.savetxt('output/' + t + '/dataset', dataset)
    np.savetxt('output/' + t + '/dataset_embedding', dataset_embedding)
    np.savetxt('output/' + t + '/new_points', new_points)
    np.savetxt('output/' + t + '/new_points embedding', new_points_embedding)


GENERATORS = [
        dataset_generator.Generator(),
        dataset_generator.Ring(width=1, radius=2),
        dataset_generator.Helix(step=0, twists=1, width=1, offset=2),
        dataset_generator.Helix(step=1, twists=1, width=1, offset=0),
        dataset_generator.Helix(step=1, twists=1, width=1, offset=2),
        dataset_generator.Helix(step=100, twists=1, width=2000, offset=211111),
        dataset_generator.Mobius(width=1, radius=1),
        dataset_generator.S_curve(),
        dataset_generator.Spiral()
]


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        g = GENERATORS[0]
        test(g=g)
    elif sys.argv[1] == 'show':
        g = GENERATORS[int(sys.argv[2])]
        test(g=g, mode="show")
    elif sys.argv[1] == 'list':
        for generator in GENERATORS:
            test(g=generator, mode="show")
