package learning4k

import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class MnistTests : Spek() { init {
    given("the Mnist data set") {
        val data = Mnist.data

        on("retrieving it") {
            it("should be properly sized") {
                assertEquals(60000, data.training.size)
                assertEquals(10000, data.test.size)
            }

            it("should contain roughly equally many samples of each digit in training data") {
                val trainingGroupedByDigit = data.training.groupBy { it.output.single() }
                assert(trainingGroupedByDigit.values.all { it.size > 5000 })
            }

            it("should contain roughly equally many samples of each digit in test data") {
                val testGroupedByDigit = data.test.groupBy { it.output.single() }
                assert(testGroupedByDigit.values.all { it.size > 800 })
            }
        }

        on("training a simple network") {
            val network = Network(
                    Layer(28 * 28, Identity),
                    Layer(10, Relu)
            )

            val cost = CrossEntropy

            fun Network.accuracy() = data.test.map {
                val prediction = network(it.input)
                val label = it.output

                val predictedDigit = prediction.indices.maxBy { prediction[it] }
                val labelDigit = label.indices.maxBy { label[it] }

                if (predictedDigit == labelDigit) 1 else 0
            }.average()

            val initialAccuracy = network.accuracy()

            val backpropagation = NumericalBackpropagation(cost = cost)

            val training = data.training.shuffle().take(10)

            val result = backpropagation.trained(network, training, learningRate = 0.0001f, steps = 1)

            val finalAccuracy = result.accuracy()

            val test = finalAccuracy
        }
    }
}
}