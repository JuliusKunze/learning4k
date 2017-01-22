package learning4k

interface Cost {
    operator fun invoke(prediction: Float, label: Float): Float

    operator fun invoke(labeledExample: LabeledExample, network: Network) =
            network(labeledExample.input).zip(labeledExample.output) { pred, target -> invoke(pred, target) }.sum()
}

object SquaredError : Cost {
    override fun invoke(prediction: Float, label: Float) = (prediction - label).let { it * it / 2 }

}

object CrossEntropy : Cost {
    override fun invoke(prediction: Float, label: Float) = -((label * log(prediction)) + (1 - label) * log(1 - prediction))
}