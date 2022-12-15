"""libraries"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn import metrics



"""configurations"""
image_size = [208,176]
batch_size = 32
learning_rate = 0.001
epochs = 12
load_model = True
train_model = False



"""load dataset"""
train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory="Alzheimer_s Dataset",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=(208,176),
    color_mode="grayscale",
    validation_split=0.2,
    subset="both",
    seed=1,
    class_names=["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
)



"""plot images of brain"""
#it plots the same pictures everytime
for count, (images, labels) in enumerate(train_dataset.take(11), start=1):
    if count == 11:
        fig, axs = plt.subplots(2,2, figsize=(7, 7))
        fig.suptitle("Alzheimer's pictures from data")
        fig.tight_layout(h_pad=4)

        axs = axs.flatten()
        for image, label, ax in zip(images, labels, axs):
            ax.set_title(train_dataset.class_names[np.argmax(label)])
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        fig.subplots_adjust(top=0.9)

        break

plt.savefig("tmp/brain.png")
plt.show()



"""load or make model"""
def load_a_model(path):
    return tf.keras.models.load_model(path)

def make_a_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(*image_size, 1)),

        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),

        tf.keras.layers.Dense(4, activation="softmax")
    ])

    return model

if load_model:
    model = load_a_model("Færdige model/model")
else:
    model = make_a_model()



"""network settings"""
model.compile(
    optimizer="adam",
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=tf.keras.metrics.CategoricalAccuracy(name='accuracy')
)



"""learning rate falloff"""
def scheduler(epoch, lr):
    decay_rate = tf.math.exp(-0.3)
    decay_step = 2
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)



"""visual representation of the model"""
model.summary()
try:
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="tmp/model.png") #Nogle gange virker funktionen, andre gange gør den ikke, ved ikke rigtig hvorfor
except:
    print("An error occurred when trying to plot model")



"""go through model and backpropagate if train_model is True"""
if train_model:
    model_history = model.fit(train_dataset, validation_data=(validation_dataset), shuffle=True, callbacks=callback, epochs=epochs)



"""temporary save model"""
model.save("tmp/save")



"""print final accuracy on training data"""
scores = model.evaluate(train_dataset)
print("Accuracy: " + str(scores[1] * 100) + "%")



"""graphs of relevant information"""
def plots(acc, val_acc, loss, val_loss, lr):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Model graphs over epochs")
    fig.tight_layout(pad=3, h_pad=2, w_pad=6)

    axs[0].plot(range(1, len(acc) + 1), acc)
    axs[0].plot(range(1, len(val_acc) + 1), val_acc)
    axs[0].set_title("Graph of accuracy over epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(["training", "validation"])

    axs[1].plot(range(1, len(loss) + 1), loss)
    axs[1].plot(range(1, len(val_loss) + 1), val_loss)
    axs[1].set_title("Graph of loss over epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend(["training", "validation"])

    axs[2].plot(range(1, len(lr) + 1), lr)
    axs[2].set_title("Graph of learning rate over epochs")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Learning rate")
    axs[2].legend(["learning rate"])

    fig.subplots_adjust(top=0.85)
    plt.savefig("tmp/graph.png")
    plt.show()



"""plot graph if model has been trained"""
if train_model:
    plots(
        model_history.history["accuracy"],    model_history.history["val_accuracy"],
        model_history.history["loss"],        model_history.history["val_loss"],
        model_history.history["lr"]
    )




"""heatmap values"""
real_labels = []
for x, y in validation_dataset.take(len(validation_dataset)):
    for label in y:
        real_labels.append(label)

prediction_labels = model.predict(validation_dataset)

real_index = np.argmax(real_labels, axis=1)
prediction_index = np.argmax(prediction_labels, axis=1)

correlation = metrics.confusion_matrix(real_index, prediction_index)



"""plot heatmap"""
plt.figure(figsize=(10, 8))
plt.tight_layout(w_pad=4, h_pad=4, pad=3)

sb.heatmap(correlation, xticklabels=validation_dataset.class_names, yticklabels=validation_dataset.class_names, annot=True, fmt="d", cmap="Blues")

plt.title("Alzheimer's diagnose prediction")
plt.ylabel("Truth")
plt.xlabel("Prediction")

plt.savefig("tmp/heatmap.png")
plt.show()