import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Network private constructor(val shape: NetworkShape, val weightsMatrices: List<WeightsMatrix>) {
    constructor(shape: NetworkShape) : this(shape = shape, weightsMatrices = shape.weightsShapes.map { WeightsMatrix(it) })

    constructor(vararg layers: Layer) : this(shape = NetworkShape(layers.toList()))

    operator fun invoke(input: List<Float>) = weightsMatrices.fold(input, { data, weight -> weight(data) })

    fun descended(gradientMatrices: List<INDArray>, learningRate: Float) =
            Network(shape, weightsMatrices.zip(gradientMatrices) { weightMatrix, gradientMatrix -> WeightsMatrix(weightMatrix.shape, weightMatrix.elements - gradientMatrix * learningRate) })

    fun copy() = Network(shape, weightsMatrices.map { it.copy() })

    override fun toString() = weightsMatrices.map { it.toString() }.joinToString()
}

class WeightsMatrix(val shape: WeightsShape, val elements: INDArray) {
    init {
        assert(shape.matrixRowCount == elements.rows())
        assert(shape.matrixColumnCount == elements.columns())
    }

    constructor(shape: WeightsShape, elements: List<List<Float>>) : this(shape, Nd4j.create(elements.map { it.toFloatArray() }.toTypedArray()))

    constructor(
            shape: WeightsShape,
            randomInitEpsilon: Double = defaultRandomInitEpsilon,
            seed: Long = System.currentTimeMillis()) :
    this(shape, Nd4j.rand(shape.matrixRowCount, shape.matrixColumnCount, -randomInitEpsilon, randomInitEpsilon, DefaultRandom(seed)))

    operator fun invoke(input: List<Float>): List<Float> {
        if (input.count() != shape.inputSize)
            throw IllegalArgumentException("${shape.inputSize} input values expected, found ${input.count()}.")

        val x = input.toColumnWithBiasUnit()
        val y = elements.transpose() * x
        return y.vectorToList().map { shape.activation(it) }
    }

    private fun List<Float>.toColumnWithBiasUnit() = (listOf(1.0f) + this).toColumnVector()

    companion object {
        val defaultRandomInitEpsilon = 0.01
    }

    fun copy() = WeightsMatrix(shape, elements = elements.dup())

    override fun toString() = elements.toString()
}