import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun main(args: Array<String>) {
    println("Test")

    val weights = Nd4j.create(listOf(1.0f, 2.0f, 3.0f, 4.0f).toFloatArray(), listOf(2, 2).toIntArray())
    val l = Layer(inputCount = 1, activation = Identity)
}

class Layer(val inputCount: Int, activation: Activation)