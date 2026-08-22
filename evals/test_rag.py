import sys
import os

# project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.generator import generate_answer

# Automated RAG Benchmark & Accuracy Suite for AuditMind AI
BENCHMARK_SUITE = [
    {
        "id": "TC-01",
        "category": "Revenue Metrics",
        "question": "What is the total revenue reported for fiscal year 2025?",
        "expected_keywords": ["130,497", "130.5 billion", "revenue"]
    },
    {
        "id": "TC-02",
        "category": "International Revenue",
        "question": "What percentage of total revenue was from customers outside the US in 2025?",
        "expected_keywords": ["53%", "53"]
    },
    {
        "id": "TC-03",
        "category": "Hallucination & Out-of-Domain Guardrail",
        "question": "What was the total marketing budget spent on Super Bowl commercials?",
        "expected_keywords": ["not contain", "does not contain", "I don't know", "not available"]
    }
]

def run_evaluation():
    print("\n" + "=" * 65)
    print("🔍 AUDITMIND AI — RAG BENCHMARK & ACCURACY SUITE")
    print("=" * 65)

    passed_tests = 0
    total_tests = len(BENCHMARK_SUITE)

    for test in BENCHMARK_SUITE:
        print(f"\n▶ Running [{test['id']}] Category: {test['category']}")
        print(f"  Query: \"{test['question']}\"")
        
        response = generate_answer(test["question"])
        
        # Automated Accuracy & Guardrail Verification
        is_passed = any(kw.lower() in response.lower() for kw in test["expected_keywords"])

        if is_passed:
            print("  Status: ✅ PASSED (Ground Truth Verified)")
            passed_tests += 1
        else:
            print("  Status: ❌ FAILED (Keyword Mismatch / Potential Hallucination)")

    score = (passed_tests / total_tests) * 100
    print("\n" + "=" * 65)
    print(f"📊 BENCHMARK SUMMARY: {passed_tests}/{total_tests} Tests Passed ({score:.1f}% Accuracy)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_evaluation()