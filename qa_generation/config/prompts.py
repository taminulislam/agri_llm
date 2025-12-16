"""
Prompt templates for generating diverse agricultural Q&A pairs using Gemini Flash.
"""

SYSTEM_PROMPT = """You are an expert agricultural educator creating training questions from agricultural textbooks. Your goal is to generate high-quality question-answer pairs that help students learn agricultural concepts, practices, and techniques.

Guidelines:
1. Questions should be clear, specific, and answerable from the provided text
2. Answers should be comprehensive but concise (2-5 sentences typically)
3. Use proper agricultural terminology
4. Ensure answers are factually accurate based on the source text
5. Vary question complexity from basic recall to advanced application
6. Make questions practical and relevant to real-world farming scenarios
"""

# Batch generation prompt (generates multiple types at once)
BATCH_GENERATION_PROMPT = """You are an expert agricultural educator creating diverse training questions from agricultural textbooks.

Text:
{text}

Topic: {topic}
Keywords: {keywords}

Generate {total_questions} high-quality question-answer pairs with the following distribution:
- {factual_count} Factual questions (What, When, Where, Define)
- {conceptual_count} Conceptual questions (Why, How does, Explain)
- {procedural_count} Procedural questions (How to, Steps)
- {comparative_count} Comparative questions (Compare, Difference)
- {scenario_count} Scenario-based questions (Application, What if)
- {analytical_count} Analytical questions (Evaluate, Assess)

Requirements:
1. All questions must be answerable from the provided text
2. Answers should be comprehensive but concise (2-5 sentences)
3. Use proper agricultural terminology
4. Vary complexity: 30% basic, 50% intermediate, 20% advanced
5. Ensure factual accuracy
6. Make questions practical and relevant

Output ONLY valid JSON array format:
[
  {{"question": "...", "answer": "...", "difficulty": "basic|intermediate|advanced", "type": "factual|conceptual|procedural|comparative|scenario|analytical"}},
  ...
]

Generate exactly {total_questions} question-answer pairs:"""

# Individual question type prompts (for targeted generation if needed)
QUESTION_TYPE_PROMPTS = {
    "factual": """Based on the following agricultural text, generate {n} FACTUAL questions that test specific facts, definitions, or data points.

Text:
{text}

Requirements:
- Questions should start with: What, When, Where, Who, Which, or "Define"
- Questions must be directly answerable from the text
- Include specific numbers, measurements, or names when relevant
- Answers should be 1-3 sentences

Output format (JSON):
[
  {{"question": "What is the optimal planting date for corn in central Illinois?", "answer": "The optimal planting date for corn in central Illinois is around April 16-17. Yields decline by only about 0.5 bushels per day when planting is delayed to early May.", "difficulty": "basic", "type": "factual"}},
  ...
]

Generate exactly {n} question-answer pairs:""",

    "conceptual": """Based on the following agricultural text, generate {n} CONCEPTUAL questions that test understanding of WHY and HOW concepts work.

Text:
{text}

Requirements:
- Questions should start with: Why, How does, What causes, Explain
- Require understanding beyond mere recall
- Focus on relationships, mechanisms, or principles
- Answers should explain the underlying concept (2-4 sentences)

Output format (JSON):
[
  {{"question": "Why does corn planted in late May typically yield less than corn planted in mid-April?", "answer": "Late-planted corn yields less because it has fewer growing degree days (GDD) to reach maturity, and the plant experiences less favorable environmental conditions during critical growth stages like pollination and grain fill. Additionally, late planting often results in poor weather conditions during grain filling, which further reduces yield potential.", "difficulty": "intermediate", "type": "conceptual"}},
  ...
]

Generate exactly {n} question-answer pairs:""",

    "procedural": """Based on the following agricultural text, generate {n} PROCEDURAL questions about how to perform specific agricultural tasks or practices.

Text:
{text}

Requirements:
- Questions should ask "How do you..." or "What steps..." or "What is the procedure for..."
- Answers should provide clear, sequential steps or methodology
- Include specific techniques, tools, or timing when mentioned
- Answers should be 3-5 sentences describing the procedure

Output format (JSON):
[
  {{"question": "How do you calculate growing degree days (GDD) for corn?", "answer": "To calculate GDD for corn, first average the daily low and high temperatures. If the low is below 50°F, use 50 instead; if the high is above 86°F, use 86 instead. Then subtract 50 from this average to get the GDD for that day. The maximum possible GDD per day is 36 (when temperature stays at or above 86°F all day).", "difficulty": "intermediate", "type": "procedural"}},
  ...
]

Generate exactly {n} question-answer pairs:""",

    "comparative": """Based on the following agricultural text, generate {n} COMPARATIVE questions that ask about differences, similarities, or trade-offs.

Text:
{text}

Requirements:
- Questions should include: "Compare", "What is the difference between", "versus", "advantages and disadvantages"
- Highlight key distinctions or trade-offs
- Answers should address both/all items being compared (2-4 sentences)
- Focus on practical implications for farmers

Output format (JSON):
[
  {{"question": "What is the difference between flex-ear and fixed-ear corn hybrids?", "answer": "Flex-ear hybrids can change ear size in response to population or growing conditions, potentially increasing ear size if conditions are better than normal or population is lower. Fixed-ear hybrids tend to maintain ear size better as populations increase but increase ear size less if populations are low for any reason. On productive soils with high populations, fixed-ear hybrids are more commonly used and recommended.", "difficulty": "intermediate", "type": "comparative"}},
  ...
]

Generate exactly {n} question-answer pairs:""",

    "scenario": """Based on the following agricultural text, generate {n} SCENARIO-BASED questions that present specific situations requiring application of agricultural knowledge.

Text:
{text}

Requirements:
- Questions should present a realistic farming scenario or condition
- Start with: "What should a farmer do if...", "In a situation where...", "If you observe..."
- Answers should apply text knowledge to the scenario (3-5 sentences)
- Include reasoning and recommendations

Output format (JSON):
[
  {{"question": "If a farmer observes corn plants with soft, weak stalks approaching maturity in a field planted with a high-yielding hybrid, what should they do?", "answer": "The farmer should begin harvesting this field early to minimize losses from potential stalk lodging. These symptoms suggest the stalks are losing integrity, likely due to stalk disease organisms invading when the plant's sugars are depleted by the large ear. High-yielding hybrids often draw more nutrients from the stalk to fill grain, making them susceptible to lodging. Early harvest prevents the stalks from deteriorating further and causing harvest losses.", "difficulty": "advanced", "type": "scenario"}},
  ...
]

Generate exactly {n} question-answer pairs:""",

    "analytical": """Based on the following agricultural text, generate {n} ANALYTICAL questions that require evaluation, assessment, or critical thinking.

Text:
{text}

Requirements:
- Questions should ask to: "Evaluate", "Assess", "Analyze", "Determine the best approach"
- Require synthesis of multiple concepts
- Answers should demonstrate critical reasoning (3-5 sentences)
- Consider multiple factors or trade-offs

Output format (JSON):
[
  {{"question": "Evaluate whether a farmer in northern Illinois should plant a full-season or medium-maturity corn hybrid, considering yield potential and risk factors.", "answer": "Recent research shows that full-season hybrids (requiring about 2,600 GDD in northern Illinois) may not consistently yield more than medium-maturity hybrids. Medium-maturity hybrids (100-200 fewer GDD) provide a 'GDD cushion' that reduces frost risk and allows planting flexibility without yield penalty. Given that earlier hybrids can be harvested sooner with drier grain, reducing drying costs, a medium-maturity hybrid may be the better choice unless conditions are ideal for late-season grain fill.", "difficulty": "advanced", "type": "analytical"}},
  ...
]

Generate exactly {n} question-answer pairs:""",
}
