package learning4k

import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class NetworkShapeTests : Spek() { init {
    given("a network architecture of 3 layers") {
        val n = NetworkShape(listOf(
                Layer(3, Identity),
                Layer(4, Relu),
                Layer(5, Relu)
        ))

        on("getting the input activation function") {
            val a = n.inputActivationFunction

            it("should be the given function") {
                assertEquals(Identity, a)
            }
        }

        on("getting the weights shapes") {
            val result = n.weightsShapes

            it("should contain two weight shapes with according sizes") {
                assertEquals(result, listOf(
                        WeightsShape(3, 4, Relu),
                        WeightsShape(4, 5, Relu)
                ))
            }
        }
    }
}
}