import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Function to initialize neurons with pre-created images from a directory
def initialize_neurons_with_images(digits, image_dir, num_neurons_per_digit):
    neurons = []
    for digit in digits:
        for filename in os.listdir(image_dir):
            if filename.startswith(f'digit_{digit}_font_'):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path)
                # array flattening
                neurons.append(np.array(image, dtype=np.float64).flatten())
                if len(neurons) >= (digit + 1) * num_neurons_per_digit:
                    # move to the next digit
                    break
                # finish viewing all the pictures
        if len(neurons) >= num_neurons_per_digit * len(digits):
            break
        # convert it to array
    neurons_array = np.array(neurons, dtype=np.float64)
    return neurons_array


# Finding the Best Matching Unit (BMU)
def find_bmu(neurons, input_vector):
    distances = np.linalg.norm(neurons - input_vector, axis=1)
    return np.argmin(distances)

# Updating neurons
def update_neurons(neurons, bmu_index, input_vector, learning_rate, radius):
    bmu = neurons[bmu_index]
    for i, neuron in enumerate(neurons):
        distance = np.linalg.norm(bmu_index - i)
        if distance <= radius:
            influence = np.exp(-distance / (2 * (radius ** 2)))
            neurons[i] += influence * learning_rate * (input_vector - neuron)
    return neurons

# Training the SOM
def train_som(data, neurons, num_epochs, initial_learning_rate, initial_radius):
    time_constant = num_epochs / np.log(initial_radius)
    for epoch in range(num_epochs):
        # learning rate and radius both decrease over time
        learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
        radius = initial_radius * np.exp(-epoch / time_constant)
        for input_vector in data:
            bmu_index = find_bmu(neurons, input_vector)
            neurons = update_neurons(neurons, bmu_index, input_vector, learning_rate, radius)
    return neurons

# Plotting the SOM
def plot_som(neurons, size):
    # there are 100 output pictures whose size is 28 X 28
    som_map = neurons.reshape(size, size, 28, 28)
    fig, ax = plt.subplots(size, size, figsize=(10, 10))
    for i in range(size):
        for j in range(size):
            ax[i, j].imshow(som_map[i, j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()

if __name__ == "__main__":
    digits = list(range(10))
    image_dir = r'C:\Users\Admin\PycharmProjects\cb-ex3\cb-ex3-git\pre_created_images'  # Directory where pre-created images are saved
    num_neurons_per_digit = 10
    total_neurons = len(digits) * num_neurons_per_digit
    size = int(np.sqrt(total_neurons))

    if size ** 2 != total_neurons:
        raise ValueError(f"Total number of neurons {total_neurons} is not a perfect square. Adjust the number of neurons per digit.")
    
    # Initialize neurons with pre-created images
    neurons = initialize_neurons_with_images(digits, image_dir, num_neurons_per_digit)
    
    # Load data and normalize
    data_path = r"C:\Users\Admin\Downloads\digits_test.csv"
    try:
        data = np.loadtxt(data_path, delimiter=',')
    except FileNotFoundError:
        print(f"File not found: {data_path}")
    else:
        # data = normalize(data, axis=1, norm='l2')
        
        num_epochs = 10
        initial_learning_rate = 0.1
        initial_radius = 5
        
        # Train the SOM
        neurons = train_som(data, neurons, num_epochs, initial_learning_rate, initial_radius)
        
        # Plot the SOM
        plot_som(neurons, size)
