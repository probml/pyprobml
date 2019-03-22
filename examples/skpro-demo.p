
#https://alan-turing-institute.github.io/skpro/introduction.html

# Load boston housing data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train and predict on boston housing data using a baseline model
y_pred = DensityBaseline().fit(X_train, y_train)\
                          .predict(X_test)
# Obtain the loss
loss = log_loss(y_test, y_pred, sample=True, return_std=True)

print('Loss: %f+-%f' % loss)
