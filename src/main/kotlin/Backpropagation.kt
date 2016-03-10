import org.nd4j.linalg.api.ndarray.INDArray

interface Backpropagation {
    val cost: Cost

    fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray>

    fun trained(network: Network, example: LabeledExample, learningRate: Float) =
            network.descended(gradientsMatrices(network, example), learningRate)

    fun trained(network: Network, examples: List<LabeledExample>, learningRate: Float) =
            examples.fold(network) { network, example -> trained(network, example, learningRate) }

    fun withValidation(validator: Backpropagation = NumericalBackpropagation(cost), tolerance: Float = 1e-4f): Backpropagation = object : Backpropagation {
        override val cost = this@Backpropagation.cost

        override fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray> {
            val result = this@Backpropagation.gradientsMatrices(network, example)
            val expectedResult = validator.gradientsMatrices(network, example)

            for ((matrixWithIndex, expectedMatrix) in result.withIndex().zip(expectedResult).reversed()) {
                val matrix = matrixWithIndex.value
                val matrixIndex = matrixWithIndex.index

                class Deviation(val index: Pair<Int, Int>) {
                    val deviation = Math.abs(matrix[index] - expectedMatrix[index])
                }

                val maxDeviation = matrix.indices.map { Deviation(it) }.maxBy { it.deviation }!!
                if (maxDeviation.deviation > tolerance) {
                    throw AssertionError("Deviation too high. First failure in layer with weight matrix $matrixIndex with maximum deviation ${maxDeviation.deviation} at matrix location ${maxDeviation.index}.")
                }
            }

            return result
        }
    }
}

class NumericalBackpropagation(cost: Cost = SquaredError, val distance: Float = 1e-5f) : Backpropagation {
    override val cost = cost

    override fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray> {
        val modified = network.copy()
        return network.weightsMatrices.mapIndexed { weightsMatrixIndex, weightsMatrix ->
            matrix(rowCount = weightsMatrix.shape.matrixRowCount, columnCount = weightsMatrix.shape.matrixColumnCount)
            { row, column ->
                val originalValue = weightsMatrix.elements[row, column]

                fun set(alternativeValue: Float) {
                    modified.weightsMatrices[weightsMatrixIndex].elements[row, column] = alternativeValue
                }

                fun cost(alternativeValue: Float): Float {
                    set(alternativeValue)
                    val cost = cost(example, modified)
                    set(originalValue)
                    return cost
                }

                (cost(originalValue + distance) - cost(originalValue - distance)) / (2 * distance)
            }
        }
    }
}

class StandardBackpropagation(cost: Cost = SquaredError) : Backpropagation {
    override val cost = cost

    override fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray> {
        val output = network.invoke(example.input)
        val prediction = example.output
        val deltaOutput = output.zip(prediction) { output, prediction -> output - prediction }

        throw NotImplementedError()
    }
}