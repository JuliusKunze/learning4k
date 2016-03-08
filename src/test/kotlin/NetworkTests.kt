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
            val m = a.weightsMatrices

            it("should be 2") {
                assert(m.count() == 2)
            }

            it("each should have only non-negative values below the max value") {
                assert(m.all { matrix -> matrix.elements.flattenToList().all { 0 <= it && it < WeightsMatrix.defaultRandomInitializationMax } })
            }

            it("none should only have 0 values") {
                assert(m.none { matrix -> matrix.elements.flattenToList().all { it == 0f } })
            }
        }
    }

    given("a weights matrix") {
        val weights = WeightsMatrix(WeightsShape(inputSize = 3, outputSize = 2, activation = Relu))

        on("invoking it") {
            val x = listOf(5f, 3f, 1f)
            val y = weights(x)

            it("should return a column of 2 values") {
                assertEquals(2, y.count())
                assert(y.all { it >= 0 })
            }
        }
    }

    given("a weights matrix with just 1 input") {
        val weights = WeightsMatrix(WeightsShape(inputSize = 1, outputSize = 3, activation = Relu), elements = listOf(
                listOf(1f, 0f, 4f),
                listOf(0f, 3f, 4f)))

        on("invoking it") {
            val y = weights(listOf(1.5f))

            it("should return 3 neurons activations") {
                assertEquals(y, listOf(1f, 3f * 1.5f, 4f + 4f * 1.5f))
            }
        }
    }

    given("a three layer network") {
        val network = Network(
                Layer(1, Identity),
                Layer(1, Relu),
                Layer(1, Relu),
                Layer(1, Relu)
        )

        val weights = listOf(
                listOf(0.5f, 3.0f),
                listOf(1.0f, 5.0f),
                listOf(2.0f, 3.0f)
        )

        weights.withIndex().forEach { network.weightsMatrices[it.index].elements.putRow(0, it.value.toRow()) }


        val x = 1.5f

        val h1 = 0.5f + x * 3.0f
        val h2 = 1.0f + h1 * 5.0f
        val expectedY = 2.0f + h2 * 3.0f

        on("invoking it") {
            val y = network(listOf(x))

            it("should return a single result") {
                assertEquals(expectedY, y.single())
            }
        }

        on("getting the squared error for the exact input/output of the network") {
            val error = SquaredError(
                    labeledData = LabeledData(listOf(x), listOf(expectedY)),
                    network = network
            )

            it("should be 0") {
                assertEquals(error, 0.0f)
            }
        }

        on("getting the squared error for the netowork output + 2") {
            val error = SquaredError(
                    labeledData = LabeledData(listOf(x), listOf(expectedY + 2)),
                    network = network
            )

            it("should be 4/2") {
                assertEquals(error, 4f / 2)
            }
        }
    }
}
}