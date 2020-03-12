import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from common.ops import ops


def dynamic_routing(votes,
                    num_routing=3,
                    softmax_in=False,
                    temper=1.0,
                    leaky=False,
                    activation='squash'):
    """ Dynamic routing algorithm.
    Args:
        votes: a tensor with shape [batch_size, ..., num_in, num_out, out_dims]
        num_routing: integer, number of routing iterations.
        softmax_in: do softmax on input capsules, default as False, which is the same as original routing
        temper: a param to make result sparser
        activation: activation of vector

    Returns:
        pose: a tensor with shape [batch_size, ..., num_out, out_dims]
        prob: a tensor with shape [batch_size, ..., num_out]
    """
    b = tf.zeros_like(votes, name='b')
    b = tf.reduce_sum(b, -1, keepdims=True)
    activation_fn = ops.get_activation(activation)
    poses = []
    probs = []
    bs = []
    cs = []

    for i in range(num_routing):
        if softmax_in:
            c = tf.nn.softmax(temper*b, axis=-3)
        else:
            if leaky:
                c = leaky_routing(temper*b, axis=-2)
            else:
                c = tf.nn.softmax(temper*b, axis=-2)

        bs.append(b)
        cs.append(c)
        pose = tf.reduce_sum(c * votes, axis=-3, keepdims=True)
        pose, prob = activation_fn(pose, axis=-1)  # get [batch_size, ..., 1, num_out, out_dim]
        poses.append(pose)
        probs.append(prob)
        distances = votes * pose
        distances = tf.reduce_sum(distances, axis=-1, keepdims=True)  # [batch_size, ..., num_in, num_out, 1]
        b += distances

    return poses, probs, bs, cs


def norm_routing(votes,
                 num_routing=3,
                 softmax_in=False,
                 temper=1.0,
                 activation='squash'):
    """ Dynamic routing algorithm.
    Args:
        votes: a tensor with shape [batch_size, ..., num_in, num_out, out_dims]
        num_routing: integer, number of routing iterations.
        softmax_in: do softmax on input capsules, default as False, which is the same as original routing
        temper: a param to make result sparser
        activation: activation of vector

    Returns:
        pose: a tensor with shape [batch_size, ..., num_out, out_dims]
        prob: a tensor with shape [batch_size, ..., num_out]
    """
    b = tf.zeros_like(votes, name='b')
    b = tf.reduce_sum(b, -1, keepdims=True)
    activation_fn = ops.get_activation(activation)
    poses = []
    probs = []
    bs = []
    cs = []
    if softmax_in:
        c = tf.nn.softmax(temper * b, axis=-3)
    else:
        c = tf.nn.softmax(temper * b, axis=-2)
    for i in range(num_routing):
        pose = tf.reduce_sum(c * votes, axis=-3, keepdims=True)
        pose, prob = ops.vector_norm(pose, axis=-1)  # get [batch_size, ..., 1, num_out, out_dim]
        distances = votes * pose
        distances = tf.reduce_sum(distances, axis=-1, keepdims=True)  # [batch_size, ..., num_in, num_out, 1]
        b += distances
        if softmax_in:
            c = tf.nn.softmax(temper*b, axis=-3)
        else:
            c = tf.nn.softmax(temper*b, axis=-2)
        bs.append(b)
        cs.append(c)

    pose = tf.reduce_sum(c * votes, axis=-3, keepdims=True)
    prob = None
    if activation_fn:
        pose, prob = activation_fn(pose, axis=-1)  # get [batch_size, ..., 1, num_out, out_dim]
    poses.append(pose)
    if prob:
        probs.append(prob)
    return poses, probs, bs, cs


def coupling_entropy(coupling, axis=-3):
    entropy = tf.reduce_sum(-coupling * tf.math.log(coupling+1e-9), axis=axis)
    return entropy


def activated_entropy(coupling, child, axis=-3):
    if child is None or len(child.get_shape().as_list()) != 3:
        child = 1
    entropy = child * coupling_entropy(coupling, axis)
    return entropy


def draw_graph(c, u_in, u_out, display=True, save='capsule.png'):
    shape = c.shape
    if len(shape) == 3:
        c = np.squeeze(c, -1)
        shape = c.shape
    in_nodes = shape[-2]
    out_nodes = shape[-1]
    X = [i for i in range(in_nodes)]
    Y = [i for i in range(in_nodes, out_nodes + in_nodes)]
    edges = []
    for i in X:
        for j in Y:
            edges.append((i, j))
    nodes = X + Y
    g = nx.Graph().to_directed()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    height_in = 10
    height_out = height_in * 0.8
    height_in_y = np.linspace(0, height_in, in_nodes)
    height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
    pos = dict()

    fig = plt.figure(figsize=(8, 3), dpi=150)
    fig.clf()
    fig.tight_layout()
    ax = fig.subplots()
    pos.update((n, (i, 1)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
    pos.update((n, (i, 2)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2

    ax.cla()
    ax.axis('off')
    node_base = 100
    u_in[u_in < 0.1] = 0.1
    u_out[u_out < 0.1] = 0.1
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='b',
                           alpha=u_in.tolist(), node_size=node_base, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='g',
                           alpha=u_out.tolist(), node_size=node_base, ax=ax)
    for edge in g.edges():
        nx.draw_networkx_edges(g, pos, edgelist=[edge], width=c[edge[0], edge[1] - in_nodes] * 1.5, ax=ax)
    if display:
        plt.show()
        plt.close()
    else:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0)
        plt.close()


def draw_anim(c, u_in, u_out, save=None):
    shape = c[0].shape
    if len(shape) == 3:
        c = np.squeeze(c, -1)
        shape = c.shape
    in_nodes = shape[-2]
    out_nodes = shape[-1]
    X = [i for i in range(in_nodes)]
    Y = [i for i in range(in_nodes, out_nodes + in_nodes)]
    edges = []
    for i in X:
        for j in Y:
            edges.append((i, j))
    nodes = X + Y
    g = nx.Graph().to_directed()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    height_in = 10
    height_out = height_in * 0.8
    height_in_y = np.linspace(0, height_in, in_nodes)
    height_out_y = np.linspace((height_in - height_out) / 2, height_out, out_nodes)
    pos = dict()

    fig = plt.figure(figsize=(8, 3), dpi=150)
    fig.clf()
    ax = fig.subplots()
    pos.update((n, (i, 1)) for i, n in zip(height_in_y, X))  # put nodes from X at x=1
    pos.update((n, (i, 2)) for i, n in zip(height_out_y, Y))  # put nodes from Y at x=2

    def weight_animate(i):
        dm = c[i]  # [36, 10]
        u_in_iter = u_in  # [36, ]
        u_out_iter = u_out[i]  # [10, ]
        node_base = 80
        # entropy = activated_entropy(dm, u_in_iter, -1).numpy()
        entropy = coupling_entropy(dm, -1).numpy()
        stds = np.std(entropy)
        means = np.mean(entropy)
        ax.cla()
        ax.axis('on')
        ax.set_title("Routing: %d  Entropy: %.3f(%.3f)" % (i+1, means, stds))

        # nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Blues)
        nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes), node_color='r',
                               alpha=u_in_iter.tolist(), node_size=node_base, ax=ax)
        nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color='b',
                               alpha=u_out_iter.tolist(), node_size=node_base, ax=ax)
        # nx.draw_networkx_nodes(g, pos, nodelist=range(in_nodes, in_nodes + out_nodes), node_color=u_out_iter.tolist(), cmap=plt.cm.Blues,
        #                        node_size=node_base, ax=ax)
        for edge in g.edges():
            nx.draw_networkx_edges(g, pos, edgelist=[edge], width=dm[edge[0], edge[1] - in_nodes] * 1.5, ax=ax)
        if save:
            fig.savefig('../../results/graph/' + save + '{}.png'.format(i + 1))

    ani2 = animation.FuncAnimation(fig, weight_animate, frames=len(c), interval=500)
    plt.show()
    plt.close()


