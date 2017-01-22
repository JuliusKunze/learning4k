package learning4k

import org.nd4j.linalg.api.ndarray.INDArray

interface Backpropagation {
    val cost: Cost

    fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray>
    fun averagedGradientsMatrices(network: Network, example: List<LabeledExample>): List<INDArray> {
        val all = example.map { gradientsMatrices(network, it) }
        return network.weightsMatrices.indices.map { index -> all.map { it[index] }.average() }
    }

    fun trained(network: Network, examples: List<LabeledExample>, learningRate: Float, steps: Int = 1) =
            (1..steps).fold(network) { network, step -> network.descended(averagedGradientsMatrices(network, examples), learningRate) }

    fun withValidation(validator: Backpropagation = NumericalBackpropagation(cost), tolerance: Float = 1e-8f): Backpropagation = object : Backpropagation {
        override val cost = this@Backpropagation.cost

        override fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray> {
            val result = this@Backpropagation.gradientsMatrices(network, example)
            val expectedResult = validator.gradientsMatrices(network, example)

            assert(result.size == expectedResult.size)

            for ((matrixWithIndex, expectedMatrix) in result.withIndex().zip(expectedResult).reversed()) {
                val matrix = matrixWithIndex.value
                val matrixIndex = matrixWithIndex.index

                class Deviation(val index: Pair<Int, Int>) {
                    val deviation = Math.abs(matrix[index] - expectedMatrix[index])
                }

                val maxDeviation = matrix.indices.map { Deviation(it) }.maxBy { it.deviation }!!
                assert(maxDeviation.deviation <= tolerance) { "Mismatch in weight matrix $matrixIndex with maximum deviation ${maxDeviation.deviation} at (row, column)=${maxDeviation.index}. Expected:\n$expectedMatrix, found\n$matrix" }
            }

            return result
        }
    }
}

class NumericalBackpropagation(override val cost: Cost = SquaredError, val epsilon: Float = 1e-4f) : Backpropagation {
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

                (cost(originalValue + epsilon) - cost(originalValue - epsilon)) / (2 * epsilon)
            }
        }
    }
}

class StandardBackpropagation(override val cost: Cost = SquaredError) : Backpropagation {
    override fun gradientsMatrices(network: Network, example: LabeledExample): List<INDArray> {
        val activations = network.activationsByLayer(example.input)
        val prediction = example.output
        val output = activations.last()
        if (output.size != prediction.size) {
            throw IllegalArgumentException("Expected example output size ${output.size}, but was ${prediction.size}.")
        }
        val outputDelta = output.zip(prediction) { output, prediction -> output - prediction }

        var outgoingDelta = outputDelta
        return network.weightsMatrices.indices.reversed().map { weightsIndex ->
            val incomingActivation = activations[weightsIndex]
            val weightsMatrix = network.weightsMatrices[weightsIndex]
            val gradientMatrix = gradientMatrix(
                    incomingActivation = incomingActivation,
                    weightsMatrix = weightsMatrix,
                    outgoingDelta = outgoingDelta
            )

            //delta not needed for input:
            if (weightsIndex != 0) {
                outgoingDelta = delta(activation = incomingActivation, outgoingWeightsMatrix = weightsMatrix, nextDelta = outgoingDelta)
            }

            gradientMatrix
        }.reversed()
    }

    fun delta(activation: List<Float>, outgoingWeightsMatrix: WeightsMatrix, nextDelta: List<Float>) =
            (outgoingWeightsMatrix.elements.transpose() * nextDelta.toColumnVector()).withoutBiasUnit().mul(activation.map { outgoingWeightsMatrix.shape.activationFunction.derivativeInvokedWithZ(it) }.toColumnVector()).vectorToList()

    fun gradientMatrix(incomingActivation: List<Float>, weightsMatrix: WeightsMatrix, outgoingDelta: List<Float>): INDArray {
        val incomingActivationWithBias = incomingActivation.withBiasUnit()

        if (weightsMatrix.shape.matrixRowCount != outgoingDelta.size) {
            throw IllegalArgumentException("Matrix row count ${weightsMatrix.shape.matrixRowCount} differs from outgoing delta size ${outgoingDelta.size}.")
        }
        if (weightsMatrix.shape.matrixColumnCount != incomingActivationWithBias.size) {
            throw IllegalArgumentException("Matrix column count ${weightsMatrix.shape.matrixColumnCount} differs from incoming activation size ${incomingActivationWithBias.size}.")
        }

        return matrix(
                rowCount = weightsMatrix.shape.matrixRowCount,
                columnCount = weightsMatrix.shape.matrixColumnCount,
                element = { row, column -> incomingActivationWithBias[column] * outgoingDelta[row] }) // TODO regularization
    }
}