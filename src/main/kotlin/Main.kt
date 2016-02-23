import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Network(val shape: NetworkArchitecture) {
    constructor(vararg layers: Layer) : this(shape = NetworkArchitecture(layers.toList()))

    val weightMatrices = shape.weightsShapes.map { WeightMatrix(it) }
}

class WeightMatrix(val shape: WeightsMatrixShape, val randomInitializationMax: Double = 0.01) {
    val elements = Nd4j.rand(shape.inputSize, shape.outputSize, 0.0, randomInitializationMax, DefaultRandom())
}