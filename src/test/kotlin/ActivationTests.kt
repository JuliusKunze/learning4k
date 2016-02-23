import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class ActivationTests : Spek() { init {
    given("an identity activation function") {
        val a = Identity

        on("invoking it") {
            val result = a(2.0f)

            it("should return the same") {
                assertEquals(result, 2.0f)
            }
        }
    }

    given("a relu activation function") {
        val a = Relu

        on("invoking it with a positive value") {
            val result = a(2.0f)

            it("should return the same") {
                assertEquals(result, 2.0f)
            }
        }

        on("invoking it with a negative value") {
            val result = a(-2.0f)

            it("should return 0") {
                assertEquals(result, 0.0f)
            }
        }
    }
}}