
import re

# Hospital aspects and words that may indicate them.
ASPECTS = {
    "Doctor": [
        "doctor", "doctors", "dr", "physician", "physicians"
    ],
    "Staff": [
        "staff", "nurse", "nurses", "receptionist",
        "reception", "employee", "employees"
    ],
    "Waiting Time": [
        "wait", "waiting", "waited", "waiting time",
        "long wait", "slow", "delay", "delayed",
        "hours", "hour", "queue", "appointment"
    ],
    "Cleanliness": [
        "clean", "dirty", "toilet", "toilets", "bathroom",
        "bathrooms", "hygiene", "cleanliness", "filthy",
        "unclean", "disgusting", "disgusted", "disgust",
        "dirt", "smell", "smelly", "stinking", "stench"
    ],
    "Facilities": [
        "facility", "facilities", "room", "rooms", "bed",
        "beds", "equipment", "building", "hospital"
    ],
    "Communication": [
        "communication", "communicate", "explained",
        "explanation", "information", "informed", "listen",
        "listened", "ignored"
    ],
    "Treatment": [
        "treatment", "care", "medicine", "medication",
        "medicines", "medications", "drug", "drugs",
        "pharmacy", "prescription", "surgery", "operation"
    ],
    "Emergency": [
        "emergency", "ambulance", "emergency department",
        "emergency room"
    ]
}

POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "friendly",
    "helpful", "kind", "professional", "clean", "quick",
    "fast", "happy", "wonderful", "best", "pleasant",
    "caring", "respectful", "efficient"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "worst", "rude", "dirty", "slow",
    "wait", "waiting", "waited", "delay", "delayed",
    "poor", "unhelpful", "unprofessional", "horrible",
    "awful", "filthy", "expensive", "painful", "unclean",
    "disrespectful", "disgusting", "disgusted", "disgust",
    "missing", "unavailable", "shortage", "empty",
    "refused", "ignored", "neglected", "broken", "smelly"
]

NEGATIVE_PHRASES = [
    "no medication", "no medications", "no medicine",
    "no medicines", "out of medication", "out of stock",
    "medication unavailable", "medicine unavailable",
    "not enough medication", "ran out of medication",
    "there was no medication", "there were no medicines",
    "hospital was disgusting", "hospital is disgusting",
    "absolutely disgusting", "very disgusting",
    "extremely disgusting", "very dirty", "extremely dirty",
    "not clean", "wasn't clean", "was not clean",
    "very rude", "extremely rude", "poor treatment",
    "bad treatment", "terrible service", "long waiting time",
    "long wait", "treated badly", "not helpful",
    "did not listen", "would not listen"
]


def _contains_keyword(text, keyword):
    """Match a keyword as a whole word or phrase."""
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def find_aspects(review):
    """Return every hospital aspect mentioned in a review."""
    text = review.lower()
    found = []

    for aspect, keywords in ASPECTS.items():
        if any(_contains_keyword(text, word) for word in keywords):
            found.append(aspect)

    # A medication shortage is also a treatment concern.
    shortage_phrases = [
        "no medication", "no medications", "no medicine",
        "no medicines", "out of medication", "out of stock",
        "medication unavailable", "medicine unavailable",
        "ran out of medication"
    ]
    if any(phrase in text for phrase in shortage_phrases):
        if "Treatment" not in found:
            found.append("Treatment")

    return found


def determine_aspect_sentiment(review, aspect):
    """Estimate sentiment around a particular aspect."""
    text = review.lower()

    if aspect not in ASPECTS:
        return "Unknown"

    keywords = ASPECTS[aspect]
    positions = []

    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        for match in re.finditer(pattern, text):
            positions.append(match.start())

    if not positions:
        return "Unknown"

    # Score the text surrounding each mention of the aspect.
    positive_count = 0
    negative_count = 0

    for position in positions:
        nearby = text[max(0, position - 100):position + 140]

        positive_count += sum(
            1 for word in POSITIVE_WORDS
            if _contains_keyword(nearby, word)
        )
        negative_count += sum(
            1 for word in NEGATIVE_WORDS
            if _contains_keyword(nearby, word)
        )

        negative_count += sum(
            2 for phrase in NEGATIVE_PHRASES
            if phrase in nearby
        )

    # Explicit shortage language means negative treatment availability.
    shortage_phrases = [
        "no medication", "no medications", "no medicine",
        "no medicines", "out of medication", "out of stock",
        "medication unavailable", "medicine unavailable",
        "not enough medication", "ran out of medication"
    ]

    if aspect == "Treatment" and any(
        phrase in text for phrase in shortage_phrases
    ):
        return "Negative"

    if negative_count > positive_count:
        return "Negative"
    if positive_count > negative_count:
        return "Positive"

    # Clear complaint language should not be reported as Unknown.
    if any(phrase in text for phrase in NEGATIVE_PHRASES):
        if aspect in find_aspects(text):
            return "Negative"

    return "Unknown"


def analyse_review(review):
    """Return a list of aspect and sentiment results."""
    results = []

    for aspect in find_aspects(review):
        results.append({
            "aspect": aspect,
            "sentiment": determine_aspect_sentiment(review, aspect)
        })

    return results


if __name__ == "__main__":
    review = input("Enter a patient review: ").strip()

    print("\nAspect Analysis:")
    results = analyse_review(review)

    if not results:
        print("No specific hospital aspects were identified.")
    else:
        for result in results:
            print(
                f"{result['aspect']}: {result['sentiment']}"
            )
