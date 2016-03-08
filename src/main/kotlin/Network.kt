import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Network(val shape: NetworkArchitecture) {
    constructor(vararg layers: Layer) : this(shape = NetworkArchitecture(layers.toList()))

    val weightMatrices = shape.weightsShapes.map { WeightsMatrix(it) }

    operator fun invoke(input: List<Float>) = weightMatrices.fold(input, { data, weight -> weight(data) })

    fun train(example: LabeledData) {
        val gradientMatrices = backpropagate(weightMatrices, example)
        //val newWeightMatrices = decent(weightMatrices, gradientMatrices, learningRate = )
    }

    private fun backpropagate(weightMatrices: List<WeightsMatrix>, example: LabeledData): List<INDArray> {
        /*val error =
        val elements = matrix()*/
        return null!!
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
        if (input.count()  != shape.inputSize)
            throw IllegalArgumentException("${shape.inputSize} input values expected, found ${input.count()}.")

        val x = input.toColumnWithBiasUnit()
        val y = elements.transpose() * x
        return y.toList().map { shape.activation(it) }
    }

    private fun List<Float>.toColumnWithBiasUnit() = (listOf(1.0f) + this).toColumn()

    companion object {
        val defaultRandomInitializationMax = 0.01
    }
}