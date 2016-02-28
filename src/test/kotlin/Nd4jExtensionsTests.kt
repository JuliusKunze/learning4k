import org.jetbrains.spek.api.Spek
import kotlin.test.assertEquals

class Nd4jExtensionsTests : Spek() {init {
    given("a list with some elements") {
        val l = listOf(1.0f, 3.0f, 2.0f)

        on("converting it to a column") {
            val c = l.toColumn()

            it("should be a column") {
                assert(c.isColumnVector)
            }

            it("should have one column") {
                assertEquals(1, c.columns())
            }

            it("should have multiple rows") {
                assertEquals(3, c.rows())
            }
        }

        on("converting it to a row") {
            val r = l.toRow()

            it("should be a row") {
                assert(r.isRowVector)
            }

            it("should have one row") {
                assertEquals(1, r.rows())
            }

            it("should have multiple column") {
                assertEquals(3, r.columns())
            }
        }
    }
}
}