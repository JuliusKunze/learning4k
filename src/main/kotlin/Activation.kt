interface Activation {
    fun invoke(value: Float): Float
}

object Identity : Activation {
    override operator fun invoke(value: Float) = value
}

object Relu : Activation {
    override operator fun invoke(value: Float) = Math.max(0.0f, value)
}

