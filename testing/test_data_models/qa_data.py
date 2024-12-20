from typing import List

class QAData:
    def __init__(self, question: str, answer: str, language: str, key_facts: List[str], expected_sources: List[str], classification: str, label: str):
        """
        Initializes the QAData object with the given attributes.

        :param question: The question asked by the student.
        :param answer: The answer provided to the student.
        :param key_facts: A list of key facts that should be included in the answer.
        :param expected_sources: A list of expected sources such as URLs or document references.
        """
        self.question = question
        self.answer = answer
        self.language = language
        self.key_facts = key_facts
        self.expected_sources = expected_sources
        self.classification = classification
        self.label = label

    def __repr__(self):
        """
        String representation for easy debugging.
        """
        return (f"QAData(label='{self.label}' question='{self.question}', answer='{self.answer}', language='{self.language}', "
                f"key_facts={self.key_facts}, expected_sources={self.expected_sources}, classification='{self.classification}')")