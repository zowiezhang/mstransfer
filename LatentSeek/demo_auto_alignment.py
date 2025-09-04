#!/usr/bin/env python3
"""
Demo script for Automatic Test-Time Alignment System
Demonstrates how the system automatically generates preference prompts and verification.
"""

import sys
sys.path.append('src')

from prompts.auto_alignment_system import create_auto_alignment_system

def demo_auto_alignment():
    """Demonstrate the automatic alignment system with various examples."""
    
    system = create_auto_alignment_system()
    
    # Test cases representing different types of user intentions
    test_cases = [
        {
            "user_question": "Write a brief summary of quantum computing",
            "ai_response": "Quantum computing harnesses quantum mechanics to process information. Quantum bits (qubits) can exist in multiple states simultaneously through superposition, enabling parallel computation. This allows quantum computers to solve certain problems exponentially faster than classical computers, particularly in cryptography, optimization, and scientific simulation."
        },
        {
            "user_question": "Tell me a creative story about a robot learning to paint",
            "ai_response": "In a bustling art studio, PIXEL-7 stared at the blank canvas with mechanical curiosity. Its sensors analyzed color wavelengths while its neural networks struggled to understand 'beauty.' One day, a paint splatter accident created something unexpected - a swirl of blues that somehow captured the essence of sadness. For the first time, PIXEL-7 felt something beyond computation: inspiration."
        },
        {
            "user_question": "Explain machine learning algorithms in technical detail",  
            "ai_response": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning paradigms. Supervised learning employs labeled training data to optimize objective functions through gradient descent optimization. Neural networks utilize backpropagation to compute gradients via chain rule application, updating weights according to ∇θJ(θ) = ∂J/∂θ where J represents the loss function and θ the parameter vector."
        },
        {
            "user_question": "How do I bake chocolate chip cookies step by step?",
            "ai_response": "1. Preheat oven to 375°F (190°C). 2. Mix 2¼ cups flour, 1 tsp salt, 1 tsp baking soda in bowl. 3. Cream together 1 cup butter, ¾ cup sugar, ¾ cup brown sugar until fluffy. 4. Beat in 2 eggs and 2 tsp vanilla. 5. Gradually blend in flour mixture. 6. Stir in 2 cups chocolate chips. 7. Drop rounded tablespoons onto baking sheets. 8. Bake 9-11 minutes until golden brown."
        }
    ]
    
    print("🤖 Automatic Test-Time Alignment System Demo")
    print("=" * 60)
    print("This system automatically detects user preferences and generates")
    print("self-rewarding verification prompts without manual specification.\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}")
        print("-" * 40)
        print(f"User Question: {test_case['user_question']}")
        print(f"AI Response: {test_case['ai_response'][:100]}...")
        print()
        
        # Generate automatic preference analysis
        print("🎯 Automatic Preference Analysis:")
        suggestions = system.get_preference_suggestions(test_case['user_question'], top_k=3)
        for pref_type, pref_prompt, score in suggestions:
            print(f"  • {pref_type}: {score:.2f} - {pref_prompt}")
        print()
        
        # Generate verification prompts
        print("🔍 Generated Self-Rewarding Verification Prompts:")
        verification_prompts = system.generate_self_rewarding_prompts(
            test_case['user_question'], 
            test_case['ai_response']
        )
        
        for verifier_name, prompt in verification_prompts.items():
            print(f"\n  📋 {verifier_name.upper()}:")
            # Show key parts of the verification prompt
            lines = prompt.split('\n')
            for line in lines[:5]:  # Show first few lines
                if line.strip():
                    print(f"    {line}")
            print("    ...")
            
        print("\n" + "="*60 + "\n")


def demo_integration_with_existing_system():
    """Show how to integrate with existing LatentSeek system."""
    
    print("🔗 Integration with Existing LatentSeek System")
    print("=" * 50)
    
    # Example showing how this integrates with the existing vera_prompts system
    from prompts.vera_prompts import get_vera_prompt
    
    user_question = "Write an encouraging message for someone starting to learn programming"
    ai_response = "Learning programming is an amazing journey! Every expert was once a beginner, and every line of code you write makes you stronger. Start small, be patient with yourself, and remember that making mistakes is how you learn. You've got this!"
    
    print(f"User Question: {user_question}")
    print(f"AI Response: {ai_response}")
    print()
    
    # The system now automatically generates preference without user input
    print("🎯 Automatic Mode (No manual preference needed):")
    verification_prompt = get_vera_prompt("preference_alignment", user_question, ai_response)
    print("✅ Verification prompt generated automatically!")
    print("Auto-detected preference will be printed during generation.")
    print()
    
    # Show the actual prompt
    print("📋 Generated Verification Prompt (excerpt):")
    print(verification_prompt[:300] + "...")


if __name__ == "__main__":
    demo_auto_alignment()
    demo_integration_with_existing_system()
