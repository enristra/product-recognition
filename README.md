# product-recognition
Dataset by Markus Klasson's [Grocery Store Dataset][marcusklasson_GroceryStoreDataset] GitHub page.

# Net 1
## Layer
 - Conv(32)
 - MaxPooling2D
 - Conv size(64)
 - MaxPooling2D
 - Flatten
 - Dense(42)
 - Dense(81) (Output layer)
## Loss
 - Adam(0.001)
 - sparse_categorical_crossentropy

# Net 2
 - Conv(32)
 - MaxPooling2D
 - Conv size(64)
 - MaxPooling2D
 - Conv size(128)
 - MaxPooling2D
 - Flatten
 - Dense(42)
 - Dense(81) (Output layer)
## Loss
 - Adam(0.001)
 - sparse_categorical_crossentropy

# Net 3
 - Conv(32, BatchNormalization, Activation('relu'), MaxPooling2D)
 - Conv(64, BatchNormalization, Activation('relu'), MaxPooling2D)
 - Conv(128, BatchNormalization, Activation('relu'), MaxPooling2D)
 - Flatten
 - Dense(42)
 - Dense(81) (Output layer)
##### Loss
 - Adam(0.01)
 - sparse_categorical_crossentropy

[marcusklasson_GroceryStoreDataset]: https://github.com/marcusklasson/GroceryStoreDataset