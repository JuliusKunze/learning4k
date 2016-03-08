import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Network private constructor(val shape: NetworkShape, val weightsMatrices: List<WeightsMatrix>) {
    constructor(shape: NetworkShape) : this(shape = shape, weightsMatrices = shape.weightsShapes.map { WeightsMatrix(it) })

    constructor(vararg layers: Layer) : this(shape = NetworkShape(layers.toList()))

    operator fun invoke(input: List<Float>) = weightsMatrices.fold(input, { data, weight -> weight(data) })

    fun descended(gradientMatrices: List<INDArray>, learningRate: Float = 1e-2f) =
            Network(shape, weightsMatrices.zip(gradientMatrices) { weightMatrix, gradientMatrix -> WeightsMatrix(weightMatrix.shape, weightMatrix.elements + gradientMatrix * learningRate) })

    fun copy() = Network(shape, weightsMatrices.map { it.copy() })
}

interface Backpropagation {
    operator fun invoke(network: Network, example: LabeledData): List<INDArray>

    fun trained(network: Network, example: LabeledData, learningRate: Float = 1e-2f): Network {
        val gradientMatrices = invoke(network, example)
        return network.descended(gradientMatrices, learningRate)
    }
}

class NumericalBackpropagation(val distance: Float = 1e-5f, val cost: Cost) : Backpropagation {
    override fun invoke(network: Network, example: LabeledData): List<INDArray> {
        val modified = network.copy()
        return network.weightsMatrices.mapIndexed { weightsMatrixIndex, weightsMatrix ->
            matrix(rowCount = weightsMatrix.shape.matrixRowCount, columnCount = weightsMatrix.shape.matrixColumnCount)
            { row, column ->
                val originalValue = weightsMatrix.elements[row, column]

                fun set(alternativeValue: Float) {
                    modified.weightsMatrices[weightsMatrixIndex].elements[row, column] = alternativeValue
                }

                fun cost(alternativeValue: Float): Float {
                    set(alternativeValue)
                    val cost = cost(example, modified)
                    set(originalValue)
                    return cost
                }

                (cost(originalValue + distance) - cost(originalValue - distance)) / (2 * distance)
            }
        }
    }
}

class StandardBackpropagation(val cost: Cost) : Backpropagation {
    override fun invoke(network: Network, example: LabeledData): List<INDArray> {
        val output = network.invoke(example.input)
        val prediction = example.output
        val deltaOutput = output.zip(prediction) { output, prediction -> output - prediction }

        throw NotImplementedError()
    }
}

class WeightsMatrix(val shape: WeightsShape, val elements: INDArray) {
    init {
        assert(shape.matrixRowCount == elements.rows())
        assert(shape.matrixColumnCount == elements.columns())
    }

    constructor(shape: WeightsShape, elements: List<List<Float>>) : this(shape, Nd4j.create(elements.map { it.toFloatArray() }.toTypedArray()))

    constructor(
            shape: WeightsShape,
            randomInitializationMax: Double = defaultRandomInitializationMax,
            seed: Long = System.currentTimeMillis()) :
    this(shape, Nd4j.rand(shape.matrixRowCount, shape.matrixColumnCount, 0.0, randomInitializationMax, DefaultRandom(seed)))

    operator fun invoke(input: List<Float>): List<Float> {
        if (input.count() != shape.inputSize)
            throw IllegalArgumentException("${shape.inputSize} input values expected, found ${input.count()}.")

        val x = input.toColumnWithBiasUnit()
        val y = elements.transpose() * x
        return y.toList().map { shape.activation(it) }
    }

    private fun List<Float>.toColumnWithBiasUnit() = (listOf(1.0f) + this).toColumn()

    companion object {
        val defaultRandomInitializationMax = 0.01
    }

    fun copy() = WeightsMatrix(shape, elements = elements.dup())
}