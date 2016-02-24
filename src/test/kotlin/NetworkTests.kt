import org.jetbrains.spek.api.Spek

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
                assert(m.all { matrix -> matrix.elements.allValues().all { 0 <= it && it < matrix.randomInitializationMax } })
            }

            it("none should only have 0 values") {
                assert(m.none { it.elements.allValues().all { it == 0.0f } })
            }
        }
    }
}
}