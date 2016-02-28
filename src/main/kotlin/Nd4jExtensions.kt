import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun List<Float>.toRow() = Nd4j.create(this.toFloatArray())
fun List<Float>.toColumn() = Nd4j.create(this.toFloatArray(), intArrayOf(size, 1))

operator fun INDArray.times(other: INDArray) = mmul(other)

fun INDArray.rowRange() = (0..rows() - 1)
fun INDArray.columnRange() = (0..columns() - 1)

fun INDArray.allColumns() = columnRange().map { getColumn(it) }
fun INDArray.allRows() = rowRange().map { getColumn(it) }

fun INDArray.flattenToList() = rowRange().flatMap { row -> columnRange().map { column -> getFloat(row, column) } }
fun INDArray.toList() = if (isColumnVector || isRowVector)
    flattenToList() else
    throw IllegalStateException("Row or column vector expected.")