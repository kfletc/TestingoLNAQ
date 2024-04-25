# main.py
# contains the function for the epoch loop training for the neural network
# trains 2 networks, one using ADAM optimizer and one using oLNAQ optimizer
# saves plots of loss and accuracy over iterations and epochs for both networks

import tensorflow as tf
import matplotlib.pyplot as plt
import mnist_reader
import utils
import optimizer
import network

def val_step(x_batch, y_batch, loss, acc, model):
    # Evaluate model on given batch of validation data
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_loss, batch_acc

def train_model(model, train_data, val_data, loss, acc, optimizer, epochs):
    train_losses_iteration, train_accs_iteration = [], []
    train_losses_epoch, train_accs_epoch = [], []
    val_losses_epoch, val_accs_epoch = [], []

    # iterate over epochs
    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_val, batch_accs_val = [], []

        # iterate over training data
        for x_batch, y_batch in train_data:
            batch_loss, batch_acc = optimizer.train_step(x_batch, y_batch, loss, acc, model)
            batch_losses_train.append(batch_loss)
            train_losses_iteration.append(batch_loss)
            batch_accs_train.append(batch_acc)
            train_accs_iteration.append(batch_acc)

        # iterate over validation data
        for x_batch, y_batch in val_data:
            batch_loss, batch_acc = val_step(x_batch, y_batch, loss, acc, model)
            batch_losses_val.append(batch_loss)
            batch_accs_val.append(batch_acc)

        # keep track of epoch level performance for training and validation data
        # and iteration level performance for training data
        train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
        val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)
        train_losses_epoch.append(train_loss)
        train_accs_epoch.append(train_acc)
        val_losses_epoch.append(val_loss)
        val_accs_epoch.append(val_acc)
        print("Epoch: " + str(epoch))
        print(f"Training loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
    return train_losses_epoch, train_accs_epoch, train_losses_iteration, train_accs_iteration, val_losses_epoch, val_accs_epoch


X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_val, y_val = mnist_reader.load_mnist('data', kind='t10k')

train_data = utils.batch_data(X_train, y_train)
val_data = utils.batch_data(X_val, y_val)

epochs = 25

cnn_network_1 = network.CNN()
adam_optimizer = optimizer.Adam()

a_train_le, a_train_ae, a_train_li, a_train_ai, a_val_le, a_val_ae = train_model(cnn_network_1, train_data, val_data,
                                                                     utils.cross_entropy_loss, utils.accuracy,
                                                                     adam_optimizer, epochs)

cnn_network_2 = network.CNN()
olnaq_optimizer = optimizer.oLNAQ()

n_train_le, n_train_ae, n_train_li, n_train_ai, n_val_le, n_val_ae = train_model(cnn_network_2, train_data, val_data,
                                                                     utils.cross_entropy_loss, utils.accuracy,
                                                                     olnaq_optimizer, epochs)

iteration_x = [x for x in range(len(a_train_li))]
epoch_x = [x for x in range(len(a_train_le))]

plt.title("Train Losses per Epoch")
plt.ylim(0, 2)
plt.plot(epoch_x, a_train_le, label='ADAM')
plt.plot(epoch_x, n_train_le, label='oLNAQ')
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('figures/TrainLossEpoch.png')

plt.clf()
plt.title("Train Accuracies per Epoch")
plt.plot(epoch_x, a_train_ae, label='ADAM')
plt.plot(epoch_x, n_train_ae, label='oLNAQ')
plt.xlabel("# of epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig('figures/TrainAccEpoch.png')

plt.clf()
plt.title("Train Losses per Iteration")
plt.ylim(0, 2)
plt.plot(iteration_x, a_train_li, label='ADAM', alpha=0.5)
plt.plot(iteration_x, n_train_li, label='oLNAQ', alpha=0.5)
plt.xlabel("# of iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('figures/TrainLossIteration.png')

plt.clf()
plt.title("Train Accuracies per Iteration")
plt.plot(iteration_x, a_train_ai, label='ADAM', alpha=0.5)
plt.plot(iteration_x, n_train_ai, label='oLNAQ', alpha=0.5)
plt.xlabel("# of iterations")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig('figures/TrainAccIteration.png')

plt.clf()
plt.title("Validation Accuracies per Epoch")
plt.plot(epoch_x, a_val_ae, label='ADAM')
plt.plot(epoch_x, n_val_ae, label='oLNAQ')
plt.xlabel("# of epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig('figures/ValAccEpoch.png')

print("\nDone.")
