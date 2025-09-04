#!/usr/bin/env python3
"""
Demo script for Preference Alignment Reward Model
Demonstrates the new reward system that works without dataset dependency.
"""

import sys
sys.path.append('src')

def demo_preference_reward_analysis():
    """Demonstrate preference analysis capabilities without requiring a model."""
    
    from rewards.reward import PreferenceAlignmentRewardModel
    
    # Create a mock reward model for analysis (without actual model loading)
    print("🎯 Preference Alignment Reward Model Demo")
    print("=" * 60)
    print("Demonstrating automatic preference detection and analysis\n")
    
    # Test cases
    test_cases = [
        {
            "question": "Write a brief summary of machine learning",
            "solution": "Machine learning enables computers to learn patterns from data without explicit programming. Key types include supervised learning (with labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). Applications range from image recognition to natural language processing."
        },
        {
            "question": "Tell me a creative story about a time-traveling cat",
            "solution": "Whiskers the cat discovered an ancient hourglass in her owner's attic. As she batted at it playfully, purple sparkles swirled around her, transporting her to ancient Egypt! There she met Cleopatra's royal cats and learned the secret of the nine lives - each life exists in a different time period. Now Whiskers travels through history, helping cats throughout the ages while keeping the timeline intact."
        },
        {
            "question": "Explain quantum computing in technical detail",
            "solution": "Quantum computing leverages quantum mechanical phenomena including superposition and entanglement. Quantum bits (qubits) exist in superposition states |0⟩ + α|1⟩, enabling parallel computation. Quantum gates perform unitary transformations on qubit states, with algorithms like Shor's factoring utilizing quantum Fourier transforms. Decoherence remains a significant challenge, requiring error correction protocols like surface codes."
        }
    ]
    
    # Create reward model instance (without actual model - just for preference analysis)
    class MockRewardModel:
        def __init__(self):
            from prompts.auto_preference_generator import create_auto_preference_generator
            self.auto_preference = True
            self.preference_generator = create_auto_preference_generator()
            
        def get_preference_analysis(self, question, detailed=False):
            if not self.auto_preference:
                return {"error": "Auto preference is disabled"}
                
            scores = self.preference_generator.analyze_user_prompt(question)
            detected = self.preference_generator.detect_dominant_preference(question)
            generated = self.preference_generator.generate_preference_prompt(question)
            
            result = {
                "detected_preference": detected,
                "generated_preference_prompt": generated,
                "confidence_scores": scores
            }
            
            if detailed:
                suggestions = self.preference_generator.generate_multiple_preferences(question, top_k=3)
                result["alternative_preferences"] = suggestions
                
            return result
    
    reward_model = MockRewardModel()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}")
        print("-" * 40)
        print(f"Question: {test_case['question']}")
        print(f"Solution: {test_case['solution'][:100]}...")
        print()
        
        # Analyze preferences
        analysis = reward_model.get_preference_analysis(test_case['question'], detailed=True)
        
        print("🎯 Preference Analysis:")
        print(f"  Detected: {analysis['detected_preference']}")
        print(f"  Generated: {analysis['generated_preference_prompt']}")
        print()
        
        print("📊 Confidence Scores:")
        for pref_type, score in analysis['confidence_scores'].items():
            if score > 0:
                print(f"  {pref_type}: {score:.2f}")
        print()
        
        if "alternative_preferences" in analysis:
            print("🔄 Alternative Preferences:")
            for pref_type, pref_prompt in analysis['alternative_preferences']:
                print(f"  {pref_type}: {pref_prompt}")
        
        print("\n" + "="*60 + "\n")


def demo_reward_system_integration():
    """Show how the new reward system integrates with existing code."""
    
    print("🔗 Integration with LatentSeek System")
    print("=" * 50)
    
    print("""
New Preference Alignment Reward Model Features:

✅ **Automatic Preference Detection**
   - No need to manually specify preferences
   - Intelligent analysis of user prompts
   - Support for 8 preference types

✅ **Dataset Independence** 
   - Works with any domain or task type
   - No GSM8K or math-specific dependencies
   - Universal preference alignment evaluation

✅ **Enhanced Verification**
   - 4 specialized preference alignment verifiers
   - Weighted scoring system
   - Normalized rewards [0, 1]

✅ **Backward Compatibility**
   - Drop-in replacement for original RewardModel
   - Legacy method support
   - Gradual migration path

✅ **Advanced Features**
   - Multi-preference evaluation
   - Detailed preference analysis
   - Confidence scoring
    """)
    
    print("\n📋 Usage Examples:")
    print("-" * 30)
    
    print("""
# Basic usage (auto-preference mode)
from rewards.reward import create_preference_reward_model

reward_model = create_preference_reward_model(model, tokenizer)
score = reward_model.get_preference_alignment_reward(question, solution)

# With manual preference specification  
score = reward_model.get_preference_alignment_reward(
    question, solution, preference="Your answer should be concise as much as possible"
)

# Multi-preference evaluation
results = reward_model.evaluate_with_multiple_preferences(question, solution, top_k=2)

# Preference analysis only
analysis = reward_model.get_preference_analysis(question, detailed=True)
    """)
    
    print("\n🎖️ Key Improvements:")
    print("-" * 30)
    print("1. 🤖 Fully automatic test-time alignment")
    print("2. 🌍 Universal domain applicability") 
    print("3. 🎯 Preference-aware reward calculation")
    print("4. 📊 Comprehensive evaluation metrics")
    print("5. 🔄 Seamless integration with existing code")


if __name__ == "__main__":
    demo_preference_reward_analysis()
    demo_reward_system_integration()
