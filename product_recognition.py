import keras.src.callbacks
import models
from plotting import plot_validation, plot_loss, plot
from tensorflow.compiler.mlir import tensorflow


def main():

    train_dataset = models.get_dataset("train.txt", True, 16)
    val_dataset = models.get_dataset("val.txt", False, 16)
    test_dataset = models.get_dataset("test.txt", False, 16)

    # model1 = models.model1()
    # print(model1.summary())
    #
    # history2 = model1.fit(train_dataset, validation_data=val_dataset, epochs=10)
    # test = model1.evaluate(test_dataset)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    # print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")

    model4 = models.model4()
    print(model4.summary())

    history2 = model4.fit(train_dataset, validation_data=val_dataset, epochs=20)
    test = model4.evaluate(test_dataset)
    plot(history2)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")



if __name__ == "__main__":
    main()