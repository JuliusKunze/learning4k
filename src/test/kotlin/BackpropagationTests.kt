import org.jetbrains.spek.api.Spek
import java.util.*
import kotlin.test.assertEquals

class BackpropagationTests : Spek() { init {
    fun randomNetwork() = Network(
            Layer(4, Identity),
            Layer(4, Relu),
            Layer(4, Relu)
    ).apply {
        weightsMatrices.forEach { it.elements.indices.forEach { indexes -> it.elements[indexes.first, indexes.second] /= WeightsMatrix.defaultRandomInitEpsilon.toFloat() } }
    }

    for (backpropagation in listOf(StandardBackpropagation().withValidation(), NumericalBackpropagation(), NumericalBackpropagation().withValidation())) {
        given("a backpropragation ($backpropagation), a random network and an example containing a generated random input and the corresponding output") {
            val network = randomNetwork()
            val exampleInput = (1..4).map { Random().nextFloat() }
            val example = LabeledExample(exampleInput, network(exampleInput))

            on("training it with the example") {
                val newNetwork = NumericalBackpropagation().trained(network, example, learningRate = 0.1f)

                it("should not change") {
                    assertEquals(0f, SquaredError(example, newNetwork))
                }
            }

            on("training the network with the example many times") {
                val newNetwork = backpropagation.trained(network, (1..100).map { example }, learningRate = 0.1f)

                it("should not change") {
                    assertEquals(0f, SquaredError(example, newNetwork))
                }
            }

            for (i in 0..200) {
                on("training the network many times with the same example") {
                    var errors = ArrayList<Float>()
                    var n = randomNetwork()
                    for (iteration in 1..100) {
                        n = backpropagation.trained(n, example, learningRate = 0.01f)
                        errors.add(SquaredError(example, n))
                    }

                    it("should have only decreasing errors") {
                        val sortedDifferences = errors.withIndex().drop(1).map { it.value - errors[it.index - 1] }
                        val maxIncrease = sortedDifferences.max()!!
                        assert(maxIncrease <= 0) { "Error increased, up to $maxIncrease. Differences sorted descending: ${sortedDifferences.sortedDescending()}." }
                    }
                }
            }

            for (i in 0..100) {
                on("training a randomly initialized network with the same example many times") {
                    val newNetwork = backpropagation.trained(randomNetwork(), (1..200).map { example }, learningRate = 0.1f)

                    it("it should have gradient matrices all zero") {
                        val gradientsMatrices = backpropagation.gradientsMatrices(newNetwork, example)
                        val tolerance = 1e-3f
                        gradientsMatrices.forEach {
                            it.flattenToList().forEach {
                                assert(-tolerance < it && it < tolerance) { "$it from $gradientsMatrices" }
                            }
                        }
                    }
                }
            }
        }
    }
}
}
