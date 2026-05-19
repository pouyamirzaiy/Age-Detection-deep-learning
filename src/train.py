import numpy as np

# image features

X_images = np.array([
    preprocess_model_image(img)
    for img in df['augmented_face']
])


# split using split column

train_indices = df[df['split'] == 'train'].index

test_indices = df[df['split'] == 'test'].index


X_images_train = X_images[train_indices]
X_images_test = X_images[test_indices]

X_tabular_train = X_tabular[train_indices]
X_tabular_test = X_tabular[test_indices]


y_train = y[train_indices]
y_test = y[test_indices]


# build model

model = build_model(
    tabular_shape=X_tabular_train.shape[1]
)


# train

model.fit(
    [X_images_train, X_tabular_train],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# evaluate

y_pred = model.predict([
    X_images_test,
    X_tabular_test
])

mae = mean_absolute_error(y_test, y_pred)

print(f'MAE: {mae}')


# save model

save_trained_model(model)
