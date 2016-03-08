import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun List<Float>.toRow() = Nd4j.create(this.toFloatArray())
fun List<Float>.toColumn() = Nd4j.create(this.toFloatArray(), intArrayOf(size, 1))

operator fun INDArray.times(other: INDArray) = mmul(other)
operator fun INDArray.times(other: Float) = mul(other)
operator fun INDArray.plus(other: INDArray) = add(other)

fun INDArray.rowRange() = (0..rows() - 1)
fun INDArray.columnRange() = (0..columns() - 1)

fun INDArray.allColumns() = columnRange().map { getColumn(it) }
fun INDArray.allRows() = rowRange().map { getColumn(it) }

fun INDArray.matrixIndices() = rowRange().flatMap { row-> columnRange().map { column -> row to column } }
operator fun INDArray.set(vararg index: Int, value: Float) = putScalar(index, value)
operator fun INDArray.get(vararg index: Int) = getFloat(index)

fun INDArray.flattenToList() = rowRange().flatMap { row -> columnRange().map { column -> getFloat(row, column) } }
fun INDArray.toList() = if (isColumnVector || isRowVector)
    flattenToList() else
    throw IllegalStateException("Row or column vector expected.")

fun matrix(rowCount: Int, columnCount: Int, element: (Int, Int) -> Float) = Nd4j.create(Array(rowCount, { row -> FloatArray(columnCount, { column -> element(row, column) }) }))