import models
from plotting import plot_validation, plot_loss


def main():
    model1 = models.model1()
    print(model1.summary())

    train_dataset = models.get_dataset("train.txt", True)
    val_dataset = models.get_dataset("val.txt", False)
    test_dataset = models.get_dataset("test.txt", False)

    history1 = model1.fit(train_dataset, validation_data=val_dataset, epochs=10)
    test = model1.evaluate(test_dataset)
    plot_validation(history1, 'Validation')
    plot_loss(history1)
    print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")



if __name__ == "__main__":
    main()