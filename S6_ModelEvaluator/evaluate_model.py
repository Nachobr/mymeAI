import numpy as np
from sklearn.metrics import mean_squared_error

# Generate predictions from the fine-tuned model
predictions = model.predict(test_data)

# Compute the mean squared error between the predictions and the original text
mse = mean_squared_error(predictions, test_labels)

# Print the mean squared error
print('Mean Squared Error:', mse)

# Compute the correlation between the predictions and the original text
corr = np.corrcoef(predictions.flatten(), test_labels.flatten())[0, 1]

# Print the correlation
print('Correlation:', corr)

# Function to generate text from the model
def generate_text(model, prompt):
    input_text = prompt
    generated_text = ''
    for i in range(100):
        x = np.array([input_text])
        predictions = model.predict(x)[0]
        index = np.argmax(predictions)
        generated_text += index_to_word[index] + ' '
        input_text = input_text[1:] + index_to_word[index]
    return generated_text

# Generate text from the model
generated_text = generate_text(model, 'Hello, how are you today?')

# Print the generated text
print('Generated Text:', generated_text)

# Compare the generated text to your original text
# You can modify this section to perform a more detailed evaluation of the model's performance
print('Original Text:', original_text)
print('Similarity:', similar(generated_text, original_text))
