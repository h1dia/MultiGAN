import matplotlib.pyplot as plt
import pandas as pd
import pylab
import seaborn as sns
import pickle
import datetime as date

from gaussian import *
from kl_divergence import *
from gan import *


def plot_gaussian():
    fig = plt.figure(figsize=(8, 8))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    df = pd.DataFrame(train_data, columns=["x", "y"])
    df2 = pd.DataFrame(generated_data, columns=["x", "y"])

    ax = fig.add_subplot(1, 1, 1)
    ax = sns.kdeplot(df2.x, df2.y, cmap="Blues", shade=True, shade_lowest=False, bw=.1, xlim=(-1, 1),
                     ylim=(-1, 1))
    pylab.plot()
    plt.show()


def save_pickles(kl_list):
    # save data
    now = date.datetime.now()
    time = "{0:%Y%m%d_%H%M%S}".format(now)

    with open("kldump_" + time + ".pickle", mode='wb') as f:
        pickle.dump(kl_list, f)
    print("END")


if __name__ == "__main__":
    sns.set_style('whitegrid')
    sns.set_context('talk')

    b_size = 200
    z_dims = 16
    gan = GAN(batch_size=b_size, z_dim=z_dims)

    g_loss_history = []
    d_loss_history = []
    kl_history = []

    with tf.Session() as sess:

        train_data = gaussian_mixture_double_circle(b_size, 8, 2, 0.4)

        # forward
        losses = gan.forward(train_data)
        # backward
        train_op = gan.train(losses)

        sess.run(tf.global_variables_initializer())
        with tf.name_scope('summary'):
            tf.summary.scalar('kl', kl_history)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./logs', sess.graph)

        for step in range(0, 10000 + 1):
            # Run network
            __, g_loss_val, d_loss_val, wd_loss_val = sess.run([train_op, losses[gan.g], losses[gan.d], losses[gan.wd]])

            g_loss_history.append(g_loss_val)
            d_loss_history.append(d_loss_val)

            if step % 100 == 0:
                generated_data = sess.run(gan.sample_data(tf.random_uniform([b_size, z_dims], minval=-1.0, maxval=1.0)
                                                          ))
                gt = train_data
                print("epoch : ", step, " ", d_loss_val, " ", g_loss_val, " ", wd_loss_val)
                kl = kl_divergence_2d(generated_data.T, train_data.T)
                kl_history.append(kl)
                print("KL : ", kl)

    sess.close()

    plt.plot( [i for i in range(0, 10001, 100)], kl_history, label="Proposed")
    plt.legend()
    plt.show()