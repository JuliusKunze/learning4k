import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class NetworkArchitectureTests : Spek() { init {
    given("a network architecture of 3 layers") {
        val a = NetworkArchitecture(listOf(
                Layer(3, Identity),
                Layer(4, Relu),
                Layer(5, Relu)
        ))

        on("getting the weights shapes") {
            val result = a.weightsShapes

            it("should contain two weight shapes with according sizes") {
                assertEquals(result, listOf(
                        WeightsMatrixShape(3, 4),
                        WeightsMatrixShape(4, 5)
                ))
            }
        }
    }
}
}