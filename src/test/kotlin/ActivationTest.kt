import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class ActivationTest : Spek() {init {
    given("an identity activation function") {
        val i = Identity

        on("invoking it") {
            val result = i(2.0f)

            it("should return the same") {
                assertEquals(result, 2.0f)
            }
        }
    }
}}