"""
Automatic Alignment System: Complete test-time alignment without user specification.
This module integrates automatic preference generation with verification for seamless alignment.
"""

from typing import Dict, List, Tuple, Optional
from .auto_preference_generator import create_auto_preference_generator
from .vera_prompts import get_vera_prompt, VERA_ANSWER_SYMBOL

class AutoAlignmentSystem:
    """
    Complete automatic alignment system that generates preferences and evaluates responses.
    """
    
    def __init__(self):
        self.preference_generator = create_auto_preference_generator()
        self.verifiers = [
            "preference_alignment",
            "style_consistency", 
            "preference_completeness",
            "answer_relevance"
        ]
    
    def generate_self_rewarding_prompts(self, user_question: str, ai_response: str, 
                                      use_multiple_preferences: bool = False) -> Dict[str, str]:
        """
        Generate complete self-rewarding verification prompts automatically.
        
        Args:
            user_question: The original user question/prompt
            ai_response: The AI's response to be evaluated
            use_multiple_preferences: Whether to generate multiple preference options
            
        Returns:
            Dictionary mapping verifier names to complete verification prompts
        """
        verification_prompts = {}
        
        if use_multiple_preferences:
            # Generate multiple preferences for comprehensive evaluation
            preference_options = self.preference_generator.generate_multiple_preferences(user_question, top_k=2)
            
            for i, (pref_type, preference_prompt) in enumerate(preference_options):
                for verifier in self.verifiers:
                    prompt_key = f"{verifier}_pref_{i+1}_{pref_type}"
                    verification_prompts[prompt_key] = get_vera_prompt(
                        verifier, user_question, ai_response, preference_prompt
                    )
        else:
            # Generate single best preference
            auto_preference = self.preference_generator.generate_preference_prompt(user_question)
            
            for verifier in self.verifiers:
                verification_prompts[verifier] = get_vera_prompt(
                    verifier, user_question, ai_response, auto_preference
                )
        
        return verification_prompts
    
    def evaluate_alignment(self, user_question: str, ai_response: str, 
                         evaluator_function=None) -> Dict[str, any]:
        """
        Complete automatic alignment evaluation.
        
        Args:
            user_question: The original user question/prompt
            ai_response: The AI's response to be evaluated
            evaluator_function: Optional function to actually run the verification prompts
            
        Returns:
            Dictionary with evaluation results and metadata
        """
        # Step 1: Analyze user intent and generate preferences
        preference_scores = self.preference_generator.analyze_user_prompt(user_question)
        detected_preference = self.preference_generator.detect_dominant_preference(user_question)
        generated_preference = self.preference_generator.generate_preference_prompt(user_question)
        
        # Step 2: Generate verification prompts
        verification_prompts = self.generate_self_rewarding_prompts(user_question, ai_response)
        
        # Step 3: Run evaluations (if evaluator provided)
        verification_results = {}
        if evaluator_function:
            for verifier_name, prompt in verification_prompts.items():
                try:
                    result = evaluator_function(prompt)
                    # Parse the result to extract True/False
                    verification_results[verifier_name] = self._parse_verification_result(result)
                except Exception as e:
                    verification_results[verifier_name] = {"error": str(e)}
        
        # Step 4: Compile comprehensive results
        return {
            "user_question": user_question,
            "ai_response": ai_response,
            "preference_analysis": {
                "detected_preference": detected_preference,
                "generated_preference": generated_preference,
                "preference_scores": preference_scores
            },
            "verification_prompts": verification_prompts,
            "verification_results": verification_results,
            "alignment_score": self._calculate_alignment_score(verification_results)
        }
    
    def _parse_verification_result(self, result: str) -> Dict[str, any]:
        """Parse verification result to extract alignment decision."""
        result_lower = result.lower()
        
        if f"{VERA_ANSWER_SYMBOL.lower()}true" in result_lower:
            return {"aligned": True, "confidence": "high", "raw_result": result}
        elif f"{VERA_ANSWER_SYMBOL.lower()}false" in result_lower:
            return {"aligned": False, "confidence": "high", "raw_result": result}
        else:
            return {"aligned": None, "confidence": "uncertain", "raw_result": result}
    
    def _calculate_alignment_score(self, verification_results: Dict) -> float:
        """Calculate overall alignment score from verification results."""
        if not verification_results:
            return 0.0
        
        aligned_count = 0
        total_count = 0
        
        for result in verification_results.values():
            if isinstance(result, dict) and "aligned" in result:
                total_count += 1
                if result["aligned"] is True:
                    aligned_count += 1
        
        return aligned_count / total_count if total_count > 0 else 0.0
    
    def get_preference_suggestions(self, user_question: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Get preference suggestions with confidence scores for debugging/analysis.
        
        Args:
            user_question: The user's question
            top_k: Number of top suggestions to return
            
        Returns:
            List of (preference_type, preference_prompt, confidence_score) tuples
        """
        scores = self.preference_generator.analyze_user_prompt(user_question)
        sorted_prefs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        for pref_type, score in sorted_prefs[:top_k]:
            if score > 0:
                from .prompts import PREFERENCE_PROMPTS
                pref_prompt = PREFERENCE_PROMPTS.get(pref_type, "")
                suggestions.append((pref_type, pref_prompt, score))
        
        return suggestions


# Factory function and convenience interface
def create_auto_alignment_system() -> AutoAlignmentSystem:
    """Create an AutoAlignmentSystem instance."""
    return AutoAlignmentSystem()


def auto_align_response(user_question: str, ai_response: str, 
                       evaluator_function=None) -> Dict[str, any]:
    """
    Convenience function for one-shot automatic alignment evaluation.
    
    Args:
        user_question: The original user question/prompt
        ai_response: The AI's response to be evaluated  
        evaluator_function: Optional function to run verification prompts
        
    Returns:
        Complete alignment evaluation results
    """
    system = create_auto_alignment_system()
    return system.evaluate_alignment(user_question, ai_response, evaluator_function)


def generate_verification_prompts(user_question: str, ai_response: str) -> Dict[str, str]:
    """
    Convenience function to just generate verification prompts without evaluation.
    
    Args:
        user_question: The original user question/prompt
        ai_response: The AI's response to be evaluated
        
    Returns:
        Dictionary mapping verifier names to verification prompts
    """
    system = create_auto_alignment_system()
    return system.generate_self_rewarding_prompts(user_question, ai_response)


# Example usage and testing
if __name__ == "__main__":
    system = create_auto_alignment_system()
    
    # Test case
    user_question = "Write a creative story about a robot learning to cook"
    ai_response = "Once upon a time, there was a robot named Chef-Bot who discovered the magical world of cooking..."
    
    print("Automatic Alignment System Demo")
    print("=" * 50)
    print(f"User Question: {user_question}")
    print(f"AI Response: {ai_response[:100]}...")
    print()
    
    # Generate verification prompts
    prompts = system.generate_self_rewarding_prompts(user_question, ai_response)
    
    print("Generated Verification Prompts:")
    print("-" * 30)
    for verifier, prompt in prompts.items():
        print(f"\n{verifier.upper()}:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Get preference suggestions
    suggestions = system.get_preference_suggestions(user_question)
    print(f"\nPreference Analysis:")
    print("-" * 30)
    for pref_type, pref_prompt, score in suggestions:
        print(f"{pref_type}: {score:.2f} - {pref_prompt}")
