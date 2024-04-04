# First, I'm importing the essential libraries that I'll need for this project.
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Here, I'm defining a class named GraphAutoencoder. This class will encapsulate everything related to the autoencoder model for graph data.
class GraphAutoencoder:
    def __init__(self, encoding_dim):
        # Upon initialization, I set the dimensionality of the encoding layer.
        self.encoding_dim = encoding_dim
        self.encoder = None  # This will be our encoder model.
        self.decoder = None  # This will hold the decoder model.
        self.autoencoder = None  # And this will be the complete autoencoder model.

    def build_autoencoder(self, num_nodes):
        # Here, I'm building the autoencoder model. It starts with defining the input layer.
        input_layer = Input(shape=(num_nodes,))
        # The encoded representation is a dense layer with a specified dimensionality and ReLU activation.
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)

        # The encoder model is defined here. It maps an input to its encoded representation.
        self.encoder = Model(input_layer, encoded)
        # I prepare the decoder model in a similar fashion.
        encoded_input = Input(shape=(self.encoding_dim,))
        decoded = Dense(num_nodes, activation='sigmoid')(encoded_input)
        self.decoder = Model(encoded_input, decoded)

        # The autoencoder maps an input to its reconstruction.
        autoencoder_output = self.decoder(self.encoder(input_layer))
        self.autoencoder = Model(input_layer, autoencoder_output)

    def compile_autoencoder(self):
        # Here, I compile the autoencoder using the Adam optimizer and binary crossentropy loss.
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    def train_autoencoder(self, adjacency_matrix, epochs, batch_size):
        # This function trains the autoencoder on the adjacency matrix of the graph.
        self.autoencoder.fit(adjacency_matrix, adjacency_matrix,
                             epochs=epochs, batch_size=batch_size, shuffle=True)

    def encode(self, adjacency_matrix):
        # Here, I encode the given adjacency matrix into a lower-dimensional representation.
        encoded_data = self.encoder.predict(adjacency_matrix)
        return encoded_data

    def decode(self, encoded_data):
        # And here, I decode the encoded data back into its original space.
        decoded_data = self.decoder.predict(encoded_data)
        return decoded_data

# The GraphVisualizer class will handle the visualization of the graphs.
class GraphVisualizer:
    def __init__(self):
        self.fig = None  # This will hold our figure.

    def visualize_graph(self, adjacency_matrix, encoded_data, decoded_data):
        # This method visualizes the original, encoded, and decoded graphs side by side.
        num_nodes = adjacency_matrix.shape[0]

        self.fig = make_subplots(rows=1, cols=3, subplot_titles=['Original Graph', 'Encoded Graph', 'Decoded Graph'])

        # Adding the edges and nodes for the original graph.
        self._add_edges(adjacency_matrix, row=1, col=1)
        self._add_nodes(list(range(num_nodes)), list(range(num_nodes)), row=1, col=1)  

        # For the encoded graph, I'm a bit creative, as the "edges" are conceptual here.
        self._add_edges(encoded_data, row=1, col=2)
        self._add_nodes(encoded_data[:, 0], encoded_data[:, 1], row=1, col=2)

        # And finally, the decoded graph, which should ideally resemble the original.
        self._add_edges(decoded_data, row=1, col=3)
        self._add_nodes(decoded_data[:, 0], decoded_data[:, 1], row=1, col=3)

        self.fig.update_layout(height=400, width=900, title_text="Graph Autoencoder Visualization")
        self.fig.show()

    # A helper function to add edges to the plot.
    def _add_edges(self, adjacency_matrix, row, col):
        edges = np.where(adjacency_matrix == 1)
        edges_trace = go.Scatter(x=edges[1], y=edges[0], mode='lines', name='Edges', line=dict(color='black'))
        self.fig.add_trace(edges_trace, row=row, col=col)

    # And another helper function to add nodes.
    def _add_nodes(self, x, y, row, col):
        nodes_trace = go.Scatter(x=x, y=y, mode='markers', name='Nodes', marker=dict(size=10))
        self.fig.add_trace(nodes_trace, row=row, col=col)

# This function simply creates a sample dataset for us to work with.
def create_sample_dataset():
    adjacency_matrix = np.array([[0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 0],
                                [0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 1],
                                [0, 0, 1, 1, 0]])
    return adjacency_matrix

# The main function orchestrates the entire process.
def main():
    adjacency_matrix = create_sample_dataset()

    # First, I create and compile the Graph Autoencoder.
    graph_autoencoder = GraphAutoencoder(encoding_dim=2)
    graph_autoencoder.build_autoencoder(num_nodes=adjacency_matrix.shape[0])
    graph_autoencoder.compile_autoencoder()

    # Then, I train the Graph Autoencoder.
    graph_autoencoder.train_autoencoder(adjacency_matrix, epochs=100, batch_size=1)

    # After training, I encode and decode the data to see how well the autoencoder has learned to compress and reconstruct the graph.
    encoded_data = graph_autoencoder.encode(adjacency_matrix)
    decoded_data = graph_autoencoder.decode(encoded_data)

    # Lastly, I visualize the original, encoded, and decoded graphs to visually assess the performance.
    graph_visualizer = GraphVisualizer()
    graph_visualizer.visualize_graph(adjacency_matrix, encoded_data, decoded_data)

# This is where the program starts.
if __name__ == '__main__':
    main()

