data class LabeledExample(val input: List<Float>, val output: List<Float>)
data class LabeledExamples(val training: List<LabeledExample>, val test: List<LabeledExample>)