import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j

class Weights(val weightsShape: WeightsShape, val randomInitializationMax: Double = 0.01) {
    val elements = Nd4j.rand(weightsShape.inputSize, weightsShape.outputSize, 0.0, randomInitializationMax, DefaultRandom())
}