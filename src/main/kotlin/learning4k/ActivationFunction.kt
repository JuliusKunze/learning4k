package learning4k

interface ActivationFunction {
    operator fun invoke(z: Float): Float
    /**
     * For any activation a = g(z) this returns g'(z) where g is the activation function.
     */
    fun derivativeInvokedWithZ(activation: Float): Float
}

object Identity : ActivationFunction {
    override operator fun invoke(z: Float) = z
    override fun derivativeInvokedWithZ(activation: Float) = 1f
}

object Relu : ActivationFunction {
    override operator fun invoke(z: Float) = Math.max(0.0f, z)
    override fun derivativeInvokedWithZ(activation: Float): Float = if (activation > 0) 1f else 0f
}

object Sigmoid : ActivationFunction {
    override operator fun invoke(z: Float) = (1 / (1 + Math.exp(-z.toDouble()))).toFloat()
    override fun derivativeInvokedWithZ(activation: Float) = activation * (1 - activation)
}