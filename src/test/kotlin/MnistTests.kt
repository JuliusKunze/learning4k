import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class MnistTests : Spek() { init {
    given("the Mnist provider") {
        val mnist = Mnist

        on("retrieving the data set") {
            val data = mnist.data

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
    }
}
}