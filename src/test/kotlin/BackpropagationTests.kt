import org.jetbrains.spek.api.Spek
import java.util.*
import kotlin.test.assertEquals

class BackpropagationTests : Spek() { init {
    given("a random network") {
        fun randomNetwork() = Network(
                Layer(4, Identity),
                Layer(4, Relu),
                Layer(4, Relu)
        ).apply {
            weightsMatrices.forEach { it.elements.matrixIndices().forEach { indexes -> it.elements[indexes.first, indexes.second] /= WeightsMatrix.defaultRandomInitEpsilon.toFloat() } }
        }

        val network = randomNetwork()
        val exampleInput = (1..4).map { Random().nextFloat() }
        val example = LabeledData(exampleInput, network(exampleInput))

        on("training it with the outcome value using numerical backpropagation") {
            val newNetwork = NumericalBackpropagation().trained(network, example, learningRate = 0.1f)

            it("should not change") {
                assertEquals(0f, SquaredError(example, newNetwork))
            }
        }

        on("training it with the outcome value using numerical backpropagation many times") {
            val newNetwork = NumericalBackpropagation().trained(network, (1..1000).map { example }, learningRate = 0.1f)

            it("should not change") {
                assertEquals(0f, SquaredError(example, newNetwork))
            }
        }

        for (i in 0..100) {
            on("training a randomly initialized network with the same example many times") {
                var errors = ArrayList<Float>()
                var n = randomNetwork()
                for (iteration in 1..100) {
                    n = NumericalBackpropagation().trained(n, example, learningRate = 0.1f)
                    errors.add(SquaredError(example, n))
                }

                it("should have only decreasing errors") {
                    for ((index, error) in errors.withIndex().drop(1)) {
                        val diffFromPrevious = errors[index - 1] - error
                        assert(diffFromPrevious > -1e-2f) { "$diffFromPrevious" }
                    }
                }
            }
        }

        for (i in 0..100) {
            on("training a randomly initialized network with the same example many times") {
                val newNetwork = NumericalBackpropagation().trained(randomNetwork(), (1..100).map { example }, learningRate = 0.1f)

                it("it should have gradient matrices all zero") {
                    val gradientsMatrices = NumericalBackpropagation().gradientsMatrices(newNetwork, example)
                    gradientsMatrices.forEach {
                        it.flattenToList().forEach {
                            assert(-1e-4f < it && it < 1e-4f) { "$it from $gradientsMatrices" }
                        }
                    }
                }
            }
        }
    }
}
}
