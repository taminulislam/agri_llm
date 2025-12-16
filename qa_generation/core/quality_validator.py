"""
Quality validation and filtering for generated Q&A pairs.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality assessment for a Q&A pair"""
    accuracy: float          # 0-10: Factual correctness
    clarity: float           # 0-10: Question clarity
    completeness: float      # 0-10: Answer completeness
    relevance: float         # 0-10: Agricultural relevance
    specificity: float       # 0-10: Question specificity
    overall: float           # Average score
    issues: List[str]        # List of detected issues
    recommendation: str      # keep, revise, or reject


class QualityValidator:
    """
    Automated quality validation for Q&A pairs.

    Implements multiple validation strategies:
    1. Heuristic checks (length, format, patterns)
    2. Quality scoring (0-10 scale)
    3. Answer-source alignment verification
    4. Diversity checking (avoid duplicates)
    """

    def __init__(self,
                 min_question_length: int = 10,
                 max_question_length: int = 300,
                 min_answer_length: int = 20,
                 max_answer_length: int = 1000,
                 min_overall_score: float = 6.0):

        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.min_overall_score = min_overall_score

        # Track seen questions for deduplication
        self.seen_questions = set()

    def heuristic_checks(self, question: str, answer: str, source_text: str) -> Tuple[bool, List[str]]:
        """
        Fast heuristic checks for obvious quality issues.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Length checks
        if len(question) < self.min_question_length:
            issues.append(f"Question too short ({len(question)} chars)")
        if len(question) > self.max_question_length:
            issues.append(f"Question too long ({len(question)} chars)")
        if len(answer) < self.min_answer_length:
            issues.append(f"Answer too short ({len(answer)} chars)")
        if len(answer) > self.max_answer_length:
            issues.append(f"Answer too long ({len(answer)} chars)")

        # Format checks
        if not question.strip().endswith('?'):
            issues.append("Question doesn't end with '?'")
        if question.strip().startswith(('Answer:', 'Q:', 'A:')):
            issues.append("Question has formatting artifacts")
        if answer.strip().startswith(('Question:', 'Q:', 'A:')):
            issues.append("Answer has formatting artifacts")

        # Content checks
        if question.lower() == answer.lower():
            issues.append("Question and answer are identical")

        # Vagueness checks
        vague_patterns = [
            r'^What is .{1,15}\?$',  # Too short
            r'^(Yes|No)\.?$',  # One-word answers
            r'^(True|False)\.?$',
        ]
        for pattern in vague_patterns:
            if re.match(pattern, question.strip(), re.IGNORECASE):
                issues.append("Question appears too vague")
            if re.match(pattern, answer.strip(), re.IGNORECASE):
                issues.append("Answer appears too simple")

        # Incomplete checks
        if '...' in question or '[' in question or '___' in question:
            issues.append("Question appears incomplete")
        if '...' in answer or '[TBD]' in answer or '[TODO]' in answer:
            issues.append("Answer appears incomplete")

        # Deduplication (fuzzy)
        question_normalized = self._normalize_text(question)
        if question_normalized in self.seen_questions:
            issues.append("Duplicate or very similar question")
        else:
            self.seen_questions.add(question_normalized)

        # Source alignment - answer should relate to source
        # Check if at least some key terms from answer appear in source
        answer_words = set(re.findall(r'\b\w{5,}\b', answer.lower()))
        source_words = set(re.findall(r'\b\w{5,}\b', source_text.lower()))
        overlap = len(answer_words & source_words)

        if overlap < 3:  # At least 3 significant words should overlap
            issues.append("Answer may not be grounded in source text")

        is_valid = len(issues) == 0
        return is_valid, issues

    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        # Remove punctuation, lowercase, remove extra spaces
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def score_quality(self, question: str, answer: str, source_text: str) -> QualityScore:
        """
        Comprehensive quality scoring.

        For now uses heuristics; can be enhanced with LLM-based scoring.
        """
        # Start with heuristic checks
        is_valid, issues = self.heuristic_checks(question, answer, source_text)

        # Initialize scores
        accuracy = 8.0  # Default, can be enhanced with LLM
        clarity = 8.0
        completeness = 8.0
        relevance = 8.0
        specificity = 8.0

        # Adjust scores based on issues
        if not is_valid:
            accuracy -= len(issues) * 0.5
            clarity -= len(issues) * 0.5
            completeness -= len(issues) * 0.5

        # Length-based adjustments
        if len(answer) < 50:
            completeness -= 2.0
        if len(answer) > 500:
            completeness -= 1.0

        # Question type detection
        question_lower = question.lower()
        if any(question_lower.startswith(w) for w in ['what', 'when', 'where', 'who']):
            specificity += 1.0  # Factual questions are specific
        if question_lower.startswith('why') or 'explain' in question_lower:
            completeness += 1.0  # Conceptual questions should be complete

        # Ensure scores are in 0-10 range
        accuracy = max(0, min(10, accuracy))
        clarity = max(0, min(10, clarity))
        completeness = max(0, min(10, completeness))
        relevance = max(0, min(10, relevance))
        specificity = max(0, min(10, specificity))

        overall = (accuracy + clarity + completeness + relevance + specificity) / 5

        # Recommendation
        if overall >= 7.5 and len(issues) == 0:
            recommendation = "keep"
        elif overall >= 6.0:
            recommendation = "revise" if issues else "keep"
        else:
            recommendation = "reject"

        return QualityScore(
            accuracy=accuracy,
            clarity=clarity,
            completeness=completeness,
            relevance=relevance,
            specificity=specificity,
            overall=overall,
            issues=issues,
            recommendation=recommendation
        )

    def filter_batch(self, qa_pairs: List[Dict], source_chunks: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a batch of Q&A pairs.

        Args:
            qa_pairs: List of {"question": ..., "answer": ..., "chunk_id": ...}
            source_chunks: Dict mapping chunk_id to source text

        Returns:
            (accepted_pairs, rejected_pairs)
        """
        accepted = []
        rejected = []

        for pair in qa_pairs:
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            chunk_id = pair.get('chunk_id', '')
            source_text = source_chunks.get(chunk_id, '')

            # Score quality
            score = self.score_quality(question, answer, source_text)

            # Add quality metadata
            pair['quality_score'] = {
                'accuracy': score.accuracy,
                'clarity': score.clarity,
                'completeness': score.completeness,
                'relevance': score.relevance,
                'specificity': score.specificity,
                'overall': score.overall,
                'issues': score.issues,
                'recommendation': score.recommendation
            }

            # Filter
            if score.recommendation == "keep" and score.overall >= self.min_overall_score:
                accepted.append(pair)
            else:
                rejected.append(pair)

        return accepted, rejected

    def calculate_diversity_metrics(self, qa_pairs: List[Dict]) -> Dict:
        """
        Calculate diversity metrics for a set of Q&A pairs.

        Returns:
            Dictionary with diversity statistics
        """
        if not qa_pairs:
            return {}

        # Question type distribution
        type_counts = {}
        for pair in qa_pairs:
            qtype = pair.get('type', 'unknown')
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        # Difficulty distribution
        diff_counts = {}
        for pair in qa_pairs:
            difficulty = pair.get('difficulty', 'unknown')
            diff_counts[difficulty] = diff_counts.get(difficulty, 0) + 1

        # Question starter diversity
        starters = {}
        for pair in qa_pairs:
            question = pair.get('question', '')
            starter = question.split()[0].lower() if question else 'unknown'
            starters[starter] = starters.get(starter, 0) + 1

        # Length statistics
        q_lengths = [len(pair.get('question', '')) for pair in qa_pairs]
        a_lengths = [len(pair.get('answer', '')) for pair in qa_pairs]

        return {
            'total_pairs': len(qa_pairs),
            'type_distribution': type_counts,
            'difficulty_distribution': diff_counts,
            'question_starters': starters,
            'avg_question_length': sum(q_lengths) / len(q_lengths) if q_lengths else 0,
            'avg_answer_length': sum(a_lengths) / len(a_lengths) if a_lengths else 0,
            'unique_question_starters': len(starters)
        }
