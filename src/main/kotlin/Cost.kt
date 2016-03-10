interface Cost {
    operator fun invoke(prediction: Float, target: Float): Float

    operator fun invoke(labeledData: LabeledExample, network: Network) =
            network(labeledData.input).zip(labeledData.output).
            map { invoke(prediction = it.first, target = it.second) }.sum ()
}

object SquaredError : Cost {
    override fun invoke(prediction: Float, target: Float) = (prediction - target).let { it * it / 2 }
}