import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class NetworkShapeTest : Spek() {init {
    given("a network of 3 layers") {
        val a = NetworkShape(listOf(
                Layer(3, Identity),
                Layer(4, Relu),
                Layer(5, Relu)
        ))

        on("getting the weights shapes") {
            val result = a.weightsShapes

            it("should contain two weight shapes with according sizes") {
                assertEquals(result, listOf(
                        WeightsShape(3, 4),
                        WeightsShape(4, 5)
                ))
            }
        }
    }
}
}