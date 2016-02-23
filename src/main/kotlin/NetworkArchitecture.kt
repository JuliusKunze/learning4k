class Layer(val size: Int, val activation: Activation)

/**
 * Represents the shape of a weight matrix between two neuron layers.
 */
data class WeightsMatrixShape(val inputSize: Int, val outputSize: Int)

data class NetworkArchitecture(val layers: List<Layer>) {
    constructor(vararg layers: Layer) : this(layers.toList())

    val weightsShapes = layers.indices.drop(1).map { WeightsMatrixShape(layers[it-1].size, layers[it].size)}
}