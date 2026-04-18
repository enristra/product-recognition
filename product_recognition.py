import models
from plotting import plot_validation, plot_loss, plot


def main():

    train_dataset = models.get_dataset("train.txt", True)
    val_dataset = models.get_dataset("val.txt", False)
    test_dataset = models.get_dataset("test.txt", False)

    # model1 = models.model1()
    # print(model1.summary())
    #
    # history2 = model1.fit(train_dataset, validation_data=val_dataset, epochs=10)
    # test = model1.evaluate(test_dataset)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    # print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")

    model3 = models.model3()
    print(model3.summary())

    history2 = model3.fit(train_dataset, validation_data=val_dataset, epochs=10)
    test = model3.evaluate(test_dataset)
    plot(history2)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")



if __name__ == "__main__":
    main()