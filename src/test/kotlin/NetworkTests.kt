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
                assert(m.none { matrix -> matrix.elements.flattenToList().all { it == 0.0f } })
            }
        }
    }

    given("a single layer") {
        val layer = Weights(WeightsShape(inputSize = 3, outputSize = 2, activation = Relu))

        on("invoking it") {
            val x = listOf(5.0f, 3.0f, 1.0f)
            val y = layer(x)

            it("should return a column of 2 values") {
                assertEquals(2, y.count())
                assert(y.all { it >= 0 })
            }
        }
    }
}
}