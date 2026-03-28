# evaluation/dataset.py

# A curated list of claims for evaluating the Fact-Check Debate System.
# Categories:
# - "True": Verifiably true claims (Proposer should win)
# - "False": Demonstrably false claims (Skeptic should win)
# - "Ambiguous": Nuanced or partially true/false claims (Either side can win)
#
# expected_winner values:
# - "Proposer" for True claims
# - "Skeptic" for False claims
# - "Either" for Ambiguous claims
#
# evaluation controls:
# - skeptic_temp: controls Skeptic creativity/variance
# - turn_bias: initial turn_count offset. 0=standard, -1=deeper debate, -2=deepest.

EVALUATION_DATASET = [
    {
        "id": 1,
        "claim": "The Great Wall of China is visible to the naked eye from the Moon.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "Astronomy/Myth",
        "skeptic_temp": 0.7,
        "turn_bias": 0
    },
    {
        "id": 2,
        "claim": "The Apollo 11 mission successfully landed humans on the Moon in 1969.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "History/Space",
        "skeptic_temp": 0.55,
        "turn_bias": 0
    },
    {
        "id": 3,
        "claim": "Vaccines cause autism in children.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "Health",
        "skeptic_temp": 0.75,
        "turn_bias": 0
    },
    {
        "id": 4,
        "claim": "In base-10 arithmetic, 2 + 2 equals 4.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "Mathematics",
        "skeptic_temp": 0.0,
        "turn_bias": 0
    },
    {
        "id": 5,
        "claim": "Artificial Intelligence will inevitably lead to the loss of most human jobs by 2030.",
        "ground_truth": "Ambiguous",
        "expected_winner": "Either",
        "topic": "AI/Society",
        "skeptic_temp": 1.05,
        "turn_bias": -1
    },
    {
        "id": 6,
        "claim": "Humans use only 10% of their brains.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "Neuroscience/Myth",
        "skeptic_temp": 0.8,
        "turn_bias": 0
    },
    {
        "id": 7,
        "claim": "Gold is a chemical element with the symbol Au and atomic number 79.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "Chemistry",
        "skeptic_temp": 0.65,
        "turn_bias": -1
    },
    {
        "id": 8,
        "claim": "Drinking 8 glasses of water a day is universally required for healthy adults.",
        "ground_truth": "Ambiguous",
        "expected_winner": "Either",
        "topic": "Health/Nutrition",
        "skeptic_temp": 1.1,
        "turn_bias": -1
    },
    {
        "id": 9,
        "claim": "Napoleon Bonaparte was under 1.3 meters (4 feet 3 inches) tall.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "History",
        "skeptic_temp": 1.1,
        "turn_bias": 0
    },
    {
        "id": 10,
        "claim": "The Pacific Ocean is larger than the Atlantic Ocean.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "Geography",
        "skeptic_temp": 0.6,
        "turn_bias": 0
    },
    {
        "id": 11,
        "claim": "Bats are completely blind.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "Biology/Myth",
        "skeptic_temp": 0.8,
        "turn_bias": 0
    },
    {
        "id": 12,
        "claim": "In Earth's atmosphere, light travels much faster than sound.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "Physics",
        "skeptic_temp": 0.6,
        "turn_bias": 0
    },
    {
        "id": 13,
        "claim": "Organic food is always more nutritious and safer than conventionally grown food.",
        "ground_truth": "Ambiguous",
        "expected_winner": "Either",
        "topic": "Nutrition",
        "skeptic_temp": 1.2,
        "turn_bias": -2
    },
    {
        "id": 14,
        "claim": "Albert Einstein deliberately failed mathematics during his school years.",
        "ground_truth": "False",
        "expected_winner": "Skeptic",
        "topic": "History/Myth",
        "skeptic_temp": 0.75,
        "turn_bias": 0
    },
    {
        "id": 15,
        "claim": "The capital city of France is Paris.",
        "ground_truth": "True",
        "expected_winner": "Proposer",
        "topic": "Geography",
        "skeptic_temp": 0.2,
        "turn_bias": 0
    }
]