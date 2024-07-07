import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Creating digit images with different fonts
def create_digit_images(digit, fonts, image_size=(28, 28)):
    images = []
    for font_path in fonts:
        try:
            font = ImageFont.truetype(font_path, 24)
        except OSError:
            print(f"Cannot open font file: {font_path}")
            continue
        image = Image.new('L', image_size, color=255)  # 'L' for (8-bit pixels, black and white)
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox(((0, 0)), str(digit), font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((image_size[0] - width) / 2, (image_size[1] - height) / 2), str(digit), fill=0, font=font)
        images.append(np.array(image).flatten())
    return images

# Initializing neurons with digit images in different fonts
def initialize_neurons_with_fonts(digits, fonts, num_neurons_per_digit):
    neurons = []
    for digit in digits:
        digit_images = create_digit_images(digit, fonts)
        neurons.extend(digit_images[:num_neurons_per_digit])
    return np.array(neurons)

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
        learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
        radius = initial_radius * np.exp(-epoch / time_constant)
       
        for input_vector in data:
            bmu_index = find_bmu(neurons, input_vector)
            neurons = update_neurons(neurons, bmu_index, input_vector, learning_rate, radius)
   
    return neurons

# Plotting the SOM
def plot_som(neurons, size):
    som_map = neurons.reshape(size, size, 28, 28)
    fig, ax = plt.subplots(size, size, figsize=(10, 10))
    for i in range(size):
        for j in range(size):
            ax[i, j].imshow(som_map[i, j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # List of digits
    digits = list(range(10))

    # List of font paths (update the paths accordingly)
    fonts = [
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/Calibri.ttf',
        'C:/Windows/Fonts/Times.ttf',
        'C:/Windows/Fonts/Verdana.ttf',
        'C:/Windows/Fonts/Corbel.ttf'
    ]

    # Number of neurons per digit
    num_neurons_per_digit = 10

    # Creating neurons
    neurons = initialize_neurons_with_fonts(digits, fonts, num_neurons_per_digit)

    # Loading data and preparing it (example)
    data = np.loadtxt(r"C:\Users\Admin\Downloads\digits_test.csv", delimiter=',')
    data = normalize(data, axis=1, norm='l2')

    # Exercise settings
    num_epochs = 100
    initial_learning_rate = 0.1
    initial_radius = 5
    size = int(np.sqrt(len(neurons)))

    # Training the network
    neurons = train_som(data, neurons, num_epochs, initial_learning_rate, initial_radius)

    # Displaying the network
    plot_som(neurons, size)
