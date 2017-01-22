package learning4k

data class Layer(val size: Int, val activationFunction: ActivationFunction)

data class WeightsShape(val inputSize: Int, val outputSize: Int, val activationFunction: ActivationFunction) {
    val matrixRowCount: Int get() = outputSize
    val matrixColumnCount: Int get() = inputSize + 1
}

data class NetworkShape private constructor(
        val inputActivationFunction: ActivationFunction = Identity,
        val weightsShapes: List<WeightsShape>) {
    constructor(layers: List<Layer>) : this(inputActivationFunction = layers.first().activationFunction,
            weightsShapes = layers.withIndex().drop(1).map
            {
                WeightsShape(
                        inputSize = layers[it.index - 1].size,
                        outputSize = it.value.size,
                        activationFunction = it.value.activationFunction)
            })

    constructor(vararg layers: Layer) : this(layers.toList())
}