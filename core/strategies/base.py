from abc import ABC, abstractmethod

class JudgeMethod(ABC):

    @abstractmethod
    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism= 'fol',
        ontology: dict | None = None,
    ) -> dict:
        """Evaluate a candidate formula against a natural language sentence.

        Args:
            nl:        Natural language sentence p.
            formula:   Candidate FOL formula φ̂.
            llm:       Optional LLM override (unused by most implementations).
            formalism: Logical formalism to use. 'fol' is default; other options may be added in the future.
            ontology:  Dict with 'relations' and 'constants' describing the
                       available predicates for this evaluation instance.
                       When provided it is included in the user turn so the
                       judge sees the ontology alongside the input.

        Returns:
            result_dict with at minimum 'quality_score', 'reasoning', 'comment', 'cost'.
        """
        pass