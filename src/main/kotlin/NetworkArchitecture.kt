data class Layer(val size: Int, val activation: Activation)

data class WeightsShape(val inputSize: Int, val outputSize: Int, val activation: Activation) {
    val matrixRowCount: Int get() = inputSize + 1
    val matrixColumnCount : Int get() = outputSize
}

data class NetworkArchitecture private constructor(
        val inputActivation: Activation = Identity,
        val weightsShapes: List<WeightsShape>) {
    constructor(layers: List<Layer>) : this(inputActivation = layers.first().activation,
            weightsShapes = layers.withIndex().drop(1).map
            {
                WeightsShape(
                        inputSize = layers[it.index - 1].size,
                        outputSize = it.value.size,
                        activation = it.value.activation)
            })

    constructor(vararg layers: Layer) : this(layers.toList())
}