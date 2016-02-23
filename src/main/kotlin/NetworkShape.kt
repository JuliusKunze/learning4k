class Layer(val size: Int, activation: Activation)

/**
 * Represents the shape of a weight matrix between two neuron layers.
 */
data class WeightsShape(val inputSize: Int, val outputSize: Int)

data class NetworkShape(val layers: List<Layer>) {
    val weightsShapes = layers.indices.drop(1).map {WeightsShape(layers[it-1].size, layers[it].size)}
}