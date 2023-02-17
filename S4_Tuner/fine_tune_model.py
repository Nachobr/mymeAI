import numpy as np
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid to search over
param_grid = {'batch_size': [32, 64, 128], 'epochs': [5, 10, 20]}

# Create the grid search object
grid = GridSearchCV(model, param_grid, cv=5)

# Fit the grid search on the training data
grid.fit(train_data, train_labels)

# Get the best hyperparameters from the grid search
best_batch_size = grid.best_params_['batch_size']
best_epochs = grid.best_params_['epochs']

# Train the model with the best hyperparameters
model.fit(train_data, train_labels, batch_size=best_batch_size, epochs=best_epochs)

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_data, val_labels)

# Print the validation loss and accuracy
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_acc)

# Save the fine-tuned model for later use
model.save('fine_tuned_language_model.h5')
