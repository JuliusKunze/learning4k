import org.nd4j.linalg.api.ndarray.INDArray

fun INDArray.rowRange() = (0..rows() - 1)
fun INDArray.columnRange() = (0..columns() - 1)

fun INDArray.allColumns() = columnRange().map { getColumn(it) }
fun INDArray.allRows() = rowRange().map { getColumn(it) }

fun INDArray.allValues() = rowRange().flatMap { row -> columnRange().map { column -> getFloat(row, column) } }