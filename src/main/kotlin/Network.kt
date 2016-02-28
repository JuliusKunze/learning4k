import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Network(val shape: NetworkArchitecture) {
    constructor(vararg layers: Layer) : this(shape = NetworkArchitecture(layers.toList()))

    val weightMatrices = shape.weightsShapes.map { Weights(it) }
}

class Weights(val shape: WeightsShape, val elements: INDArray) {
    init {
        assert(shape.outputSize == elements.columns())
        assert(shape.inputSize + 1 == elements.rows())
    }

    constructor(
            shape: WeightsShape,
            randomInitializationMax: Double = defaultRandomInitializationMax,
            seed: Long = System.currentTimeMillis()) :
    this(shape, Nd4j.rand(shape.inputSize + 1, shape.outputSize, 0.0, randomInitializationMax, DefaultRandom(seed)))

    operator fun invoke(input: List<Float>) = (elements.transpose() * ((listOf(1.0f) + input).toColumn())).toList().map { shape.activation(it) }

    companion object {
        val defaultRandomInitializationMax = 0.01
    }
}