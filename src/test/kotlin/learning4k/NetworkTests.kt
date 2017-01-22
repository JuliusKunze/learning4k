package learning4k

import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class NetworkTests : Spek() { init {
    given("a network with 3 layers") {
        val a = Network(
                Layer(3, Identity),
                Layer(4, Sigmoid),
                Layer(5, Sigmoid)
        )

        on("getting the weight matrices") {
            val m = a.weightsMatrices

            it("should be 2") {
                assert(m.count() == 2)
            }

            it("each should only have a value between epsilon and -epsilon") {
                assert(m.all { matrix -> matrix.elements.flattenToList().all { -WeightsMatrix.defaultRandomInitEpsilon <= it && it < WeightsMatrix.defaultRandomInitEpsilon } })
            }

            it("none should only have 0 values") {
                assert(m.none { matrix -> matrix.elements.flattenToList().all { it == 0f } })
            }
        }
    }

    given("a weights matrix") {
        val weights = WeightsMatrix(WeightsShape(inputSize = 3, outputSize = 2, activationFunction = Relu))

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
        val weights = WeightsMatrix(WeightsShape(inputSize = 1, outputSize = 3, activationFunction = Relu), elements = listOf(
                listOf(1f, 0f),
                listOf(0f, 3f),
                listOf(4f, 4f)))

        on("invoking it") {
            val y = weights(listOf(1.5f))

            it("should return 3 neurons activations") {
                assertEquals(y, listOf(1f, 3f * 1.5f, 4f + 4f * 1.5f))
            }
        }
    }

    given("a three layer network") {
        val x = 1.5f
        val h1 = 0.5f + x * 3.0f
        val h2 = 1.0f + h1 * 5.0f
        val expectedY = 2.0f + h2 * 3.0f
        val example = LabeledExample(listOf(x), listOf(expectedY))

        fun untrainedNetwork(): Network {
            return Network(
                    Layer(1, Identity),
                    Layer(1, Relu),
                    Layer(1, Relu),
                    Layer(1, Relu)
            )
        }

        fun network(): Network {
            val network = untrainedNetwork()

            val weights = listOf(
                    listOf(0.5f, 3.0f),
                    listOf(1.0f, 5.0f),
                    listOf(2.0f, 3.0f)
            )

            weights.withIndex().forEach { network.weightsMatrices[it.index].elements.putRow(0, it.value.toRowVector()) }
            return network
        }

        on("invoking it on an example") {
            val y = network()(example.input)

            it("should return the expected result") {
                assertEquals(example.output, y)
            }
        }

        on("getting the squared error for the exact input/output of the network") {
            val error = SquaredError(example, network())

            it("should be 0") {
                assertEquals(error, 0.0f)
            }
        }

        on("getting the squared error for the network output + 2") {
            val error = SquaredError(LabeledExample(listOf(x), listOf(expectedY + 2)), network())

            it("should be 4/2") {
                assertEquals(error, 4f / 2)
            }
        }
    }
}
}