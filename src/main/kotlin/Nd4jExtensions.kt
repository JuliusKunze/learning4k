import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun List<Float>.toRowVector() = Nd4j.create(this.toFloatArray())
fun List<Float>.toColumnVector() = Nd4j.create(this.toFloatArray(), intArrayOf(size, 1))

operator fun INDArray.times(other: INDArray) = mmul(other)
operator fun INDArray.times(other: Float) = mul(other)
operator fun INDArray.plus(other: INDArray) = add(other)
operator fun INDArray.minus(other: INDArray) = sub(other)
val INDArray.rowRange: IntRange get() = 0..rows() - 1
val INDArray.columnRange: IntRange get() = 0..columns() - 1
val INDArray.indices: List<Pair<Int, Int>> get() = rowRange.flatMap { row -> columnRange.map { column -> row to column } }

operator fun INDArray.set(vararg index: Int, value: Float) = putScalar(index, value)
operator fun INDArray.get(vararg index: Int) = getFloat(index)
operator fun INDArray.set(index: Pair<Int, Int>, value: Float) = set(*(index.toList().toIntArray()), value = value)
operator fun INDArray.get(index: Pair<Int, Int>): Float = get(*(index.toList().toIntArray()))

fun INDArray.flattenToList() = indices.map { this[it] }
fun INDArray.vectorToList() = if (isColumnVector || isRowVector)
    flattenToList() else
    throw IllegalStateException("Row or column vector expected.")

fun matrix(rowCount: Int, columnCount: Int, element: (Int, Int) -> Float) = Nd4j.create(Array(rowCount, { row -> FloatArray(columnCount, { column -> element(row, column) }) }))

fun List<Float>.withBiasUnit() = listOf(1.0f) + this
fun INDArray.withoutBiasUnit() = vectorToList().drop(1).toColumnVector()