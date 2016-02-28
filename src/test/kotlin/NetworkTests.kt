import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class NetworkTests : Spek() { init {
    given("a network with 3 layers") {
        val a = Network(
                Layer(3, Identity),
                Layer(4, Relu),
                Layer(5, Relu)
        )

        on("getting the weight matrices") {
            val m = a.weightMatrices

            it("should be 2") {
                assert(m.count() == 2)
            }

            it("each should have only non-negative values below the max value") {
                assert(m.all { matrix -> matrix.elements.flattenToList().all { 0 <= it && it < Weights.defaultRandomInitializationMax } })
            }

            it("none should only have 0 values") {
                assert(m.none { matrix -> matrix.elements.flattenToList().all { it == 0f } })
            }
        }
    }

    given("a weights matrix") {
        val weights = Weights(WeightsShape(inputSize = 3, outputSize = 2, activation = Relu))

        on("invoking it") {
            val x = listOf(5f, 3f, 1f)
            val y = weights(x)

            it("should return a column of 2 values") {
                assertEquals(2, y.count())
                assert(y.all { it >= 0 })
            }
        }
    }

    given("a tiny weights matrix") {
        val weights = Weights(WeightsShape(inputSize = 1, outputSize = 3, activation = Relu), elements = listOf(
                listOf(1f, 0f, 4f),
                listOf(0f, 3f, 4f)))

        on("invoking it") {
            val y = weights(listOf(1.5f))

            it("should return three neurons activations") {
                assertEquals(y, listOf(1f, 3f * 1.5f, 4f + 4f * 1.5f))
            }
        }
    }
}
}