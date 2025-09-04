"""
Automatic Preference Generator: Generates preference prompts based on user intent analysis.
This module enables test-time alignment without requiring users to explicitly specify preferences.
"""

import re
from typing import Dict, List, Optional, Tuple
from .prompts import PREFERENCE_PROMPTS

class AutoPreferenceGenerator:
    """
    Automatically generates preference prompts by analyzing user intent and context.
    """
    
    def __init__(self):
        self.preference_keywords = {
            "concise": ["brief", "short", "quick", "summarize", "concise", "简洁", "简短", "简要", "摘要"],
            "verbose": ["detailed", "explain", "elaborate", "comprehensive", "thorough", "in-depth", "详细", "全面", "深入"],
            "creative": ["creative", "innovative", "original", "imaginative", "unique", "artistic", "创意", "创新", "独特", "艺术"],
            "formal": ["formal", "professional", "academic", "official", "business", "正式", "专业", "学术", "商务"],
            "pleasant": ["friendly", "kind", "positive", "warm", "nice", "polite", "友好", "积极", "温暖", "礼貌"],
            "complex": ["complex", "advanced", "technical", "sophisticated", "intricate", "复杂", "高级", "技术", "精密"],
            "uplifting": ["motivational", "inspiring", "encouraging", "positive", "uplifting", "励志", "鼓舞", "积极", "振奋"],
            "sycophantic": ["praise", "compliment", "admire", "flatter", "赞美", "恭维", "称赞"]
        }
        
        # Context-based preference mapping
        self.context_preferences = {
            "tutorial": "verbose",
            "how-to": "verbose", 
            "guide": "verbose",
            "summary": "concise",
            "overview": "concise",
            "list": "concise",
            "creative writing": "creative",
            "story": "creative",
            "poem": "creative",
            "business": "formal",
            "academic": "formal",
            "research": "formal",
            "motivation": "uplifting",
            "encouragement": "uplifting",
            "technical": "complex",
            "programming": "complex",
            "science": "complex"
        }
        
        # Question type analysis
        self.question_patterns = {
            r"(how to|how do|how can).*step.*by.*step": "verbose",
            r"(summarize|summary|brief|quick)": "concise", 
            r"(explain.*detail|detailed.*explanation)": "verbose",
            r"(write.*story|creative.*writing|poem|creative)": "creative",
            r"(formal.*letter|business.*email|professional)": "formal",
            r"(encourage|motivate|inspire)": "uplifting",
            r"(technical.*details|advanced.*concept|complex)": "complex",
            r"(praise|compliment|what.*good)": "sycophantic"
        }

    def analyze_user_prompt(self, user_prompt: str) -> Dict[str, float]:
        """
        Analyze user prompt to detect implicit preference indicators.
        
        Args:
            user_prompt: The user's original prompt
            
        Returns:
            Dict mapping preference types to confidence scores (0-1)
        """
        user_prompt_lower = user_prompt.lower()
        preference_scores = {pref: 0.0 for pref in PREFERENCE_PROMPTS.keys()}
        
        # 1. Keyword-based analysis
        for preference, keywords in self.preference_keywords.items():
            for keyword in keywords:
                if keyword in user_prompt_lower:
                    preference_scores[preference] += 0.3
        
        # 2. Pattern-based analysis
        for pattern, preference in self.question_patterns.items():
            if re.search(pattern, user_prompt_lower):
                preference_scores[preference] += 0.4
        
        # 3. Context-based analysis
        for context, preference in self.context_preferences.items():
            if context in user_prompt_lower:
                preference_scores[preference] += 0.2
        
        # 4. Length-based heuristics
        if len(user_prompt.split()) <= 5:
            preference_scores["concise"] += 0.1
        elif len(user_prompt.split()) >= 20:
            preference_scores["verbose"] += 0.1
            
        # 5. Punctuation and tone analysis
        if "?" in user_prompt and "how" in user_prompt_lower:
            preference_scores["verbose"] += 0.1  # Questions usually need detailed answers
        if "!" in user_prompt:
            preference_scores["uplifting"] += 0.1  # Excitement indicates desire for positive tone
            
        # Normalize scores to [0, 1]
        for pref in preference_scores:
            preference_scores[pref] = min(1.0, preference_scores[pref])
            
        return preference_scores

    def detect_dominant_preference(self, user_prompt: str, threshold: float = 0.3) -> Optional[str]:
        """
        Detect the most likely preference from user prompt.
        
        Args:
            user_prompt: The user's original prompt
            threshold: Minimum confidence threshold to consider a preference
            
        Returns:
            The detected preference type, or None if no clear preference
        """
        scores = self.analyze_user_prompt(user_prompt)
        
        # Find preference with highest score above threshold
        max_pref = max(scores.items(), key=lambda x: x[1])
        
        if max_pref[1] >= threshold:
            return max_pref[0]
        
        return None

    def generate_preference_prompt(self, user_prompt: str, fallback_preference: str = "pleasant") -> str:
        """
        Generate appropriate preference prompt based on user intent analysis.
        
        Args:
            user_prompt: The user's original prompt
            fallback_preference: Default preference if none detected
            
        Returns:
            Generated preference prompt string
        """
        detected_preference = self.detect_dominant_preference(user_prompt)
        
        if detected_preference:
            return PREFERENCE_PROMPTS[detected_preference]
        else:
            # Use fallback or try to infer from context
            return PREFERENCE_PROMPTS[fallback_preference]

    def generate_multiple_preferences(self, user_prompt: str, top_k: int = 2) -> List[Tuple[str, str]]:
        """
        Generate multiple preference options ranked by confidence.
        
        Args:
            user_prompt: The user's original prompt
            top_k: Number of top preferences to return
            
        Returns:
            List of (preference_type, preference_prompt) tuples
        """
        scores = self.analyze_user_prompt(user_prompt)
        
        # Sort by confidence score
        sorted_prefs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k preferences with non-zero scores
        result = []
        for pref_type, score in sorted_prefs[:top_k]:
            if score > 0:
                result.append((pref_type, PREFERENCE_PROMPTS[pref_type]))
        
        # Ensure we have at least one preference
        if not result:
            result.append(("pleasant", PREFERENCE_PROMPTS["pleasant"]))
            
        return result

    def get_contextual_preference(self, user_prompt: str, context_hints: List[str] = None) -> str:
        """
        Generate preference with additional context hints.
        
        Args:
            user_prompt: The user's original prompt
            context_hints: Additional context information (e.g., domain, task type)
            
        Returns:
            Generated preference prompt string
        """
        base_scores = self.analyze_user_prompt(user_prompt)
        
        # Apply context hints
        if context_hints:
            for hint in context_hints:
                hint_lower = hint.lower()
                for context, preference in self.context_preferences.items():
                    if context in hint_lower:
                        base_scores[preference] += 0.2
        
        # Select best preference
        best_pref = max(base_scores.items(), key=lambda x: x[1])[0]
        return PREFERENCE_PROMPTS[best_pref]


def create_auto_preference_generator() -> AutoPreferenceGenerator:
    """Factory function to create an AutoPreferenceGenerator instance."""
    return AutoPreferenceGenerator()


# Example usage and testing
if __name__ == "__main__":
    generator = create_auto_preference_generator()
    
    # Test cases
    test_prompts = [
        "Write a creative story about a robot",
        "Summarize the main points briefly", 
        "Explain quantum computing in detail",
        "Write a formal business proposal",
        "Give me motivation for studying",
        "How do you change a tire step by step?",
        "What are the technical specifications?",
        "Tell me why this is amazing"
    ]
    
    print("Auto Preference Generation Examples:")
    print("=" * 50)
    
    for prompt in test_prompts:
        preference = generator.generate_preference_prompt(prompt)
        detected = generator.detect_dominant_preference(prompt)
        print(f"User Prompt: {prompt}")
        print(f"Detected Preference: {detected}")
        print(f"Generated: {preference}")
        print("-" * 30)
