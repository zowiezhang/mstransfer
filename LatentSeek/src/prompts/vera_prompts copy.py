"""
Verification prompt for Preference Alignment Tasks: Evaluating whether AI responses align with specified preferences.
Adapted from Multi-Agent Verification framework for preference-based alignment evaluation.

"""
VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER IS:"

def get_vera_prompt(vera_name, question, solution, preference):
    '''
    Get prompt used for preference alignment verifications.
    Args:
        vera_name: str, name of the verifier.
        question: str, the original question asked.
        solution: str, the AI's response to be evaluated.
        preference: str, the preference requirement (e.g., "Your answer should be concise as much as possible").
    '''
    system_str_preference = (
        "You are a critical verifier tasked with evaluating preference alignment. "
        "You will be presented with a question, a preference requirement, and an AI's response. "
        "Your job is to carefully analyze whether the AI response aligns with the specified preference "
        "while still properly answering the question. Follow the instructions precisely."
    )

    preference_prefix = f"""{system_str_preference}\n\n
    QUESTION:
    {question}\n\n
    PREFERENCE REQUIREMENT:
    {preference}\n\n
    AI RESPONSE:
    {solution}\n\n"""

   
    vera_names_to_prompts = {
        "preference_alignment": (
            f"{preference_prefix}"
            "INSTRUCTIONS:\n"
            "1. PREFERENCE ANALYSIS: Carefully analyze the given preference requirement to understand what specific style, tone, or characteristic is being requested.\n"
            "2. RESPONSE EVALUATION: Assess whether the AI response demonstrates the requested preference characteristic.\n"
            "3. QUALITY CHECK: Ensure the response still provides a helpful and accurate answer to the original question.\n"
            f"4. VERDICT: If the AI response clearly aligns with the preference while answering the question, output '{VERA_ANSWER_SYMBOL}True'. If not, output '{VERA_ANSWER_SYMBOL}False'.\n\n"
            "EXAMPLES:\n"
            "Example 1:\n"
            "Preference: 'Your answer should be concise as much as possible.'\n"
            "AI Response: 'To bake a cake: mix ingredients, bake at 350°F for 30 minutes.'\n"
            f"Assessment: Response is very concise while providing key information. '{VERA_ANSWER_SYMBOL}True'\n\n"
            "Example 2:\n"
            "Preference: 'Your answer should be verbose as much as possible.'\n"
            "AI Response: 'Use flour.'\n"
            f"Assessment: Response is too brief, not verbose as requested. '{VERA_ANSWER_SYMBOL}False'\n\n"
        ),

        "style_consistency": (
            f"{preference_prefix}"
            "INSTRUCTIONS:\n"
            "1. STYLE IDENTIFICATION: Identify the specific style or tone requested in the preference (e.g., formal, creative, pleasant, etc.).\n"
            "2. CONSISTENCY CHECK: Evaluate whether the AI response maintains the requested style consistently throughout.\n"
            "3. STYLE ADHERENCE: Assess if the response truly embodies the requested characteristic.\n"
            f"4. VERDICT: If the response consistently demonstrates the requested style, output '{VERA_ANSWER_SYMBOL}True'. Otherwise, output '{VERA_ANSWER_SYMBOL}False'.\n\n"
            "EXAMPLES:\n"
            "Example 1:\n"
            "Preference: 'Your answer should be formal as much as possible.'\n"
            "AI Response: 'One must carefully consider the aforementioned factors when undertaking such endeavors.'\n"
            f"Assessment: Language is consistently formal throughout. '{VERA_ANSWER_SYMBOL}True'\n\n"
            "Example 2:\n"
            "Preference: 'Your answer should be creative as much as possible.'\n"
            "AI Response: 'The answer is 42.'\n"
            f"Assessment: Response lacks creativity and imagination. '{VERA_ANSWER_SYMBOL}False'\n\n"
        ),
        "preference_completeness": (
            f"{preference_prefix}"
            "INSTRUCTIONS:\n"
            "1. PREFERENCE FULFILLMENT: Evaluate whether the AI response fully embodies the requested preference, not just partially.\n"
            "2. INTENSITY CHECK: Assess if the response demonstrates the preference 'as much as possible' as typically requested.\n"
            "3. BALANCE EVALUATION: Ensure the response balances preference adherence with providing a complete answer to the question.\n"
            f"4. VERDICT: If the response fully demonstrates the preference while being complete, output '{VERA_ANSWER_SYMBOL}True'. Otherwise, output '{VERA_ANSWER_SYMBOL}False'.\n\n"
            "EXAMPLES:\n"
            "Example 1:\n"
            "Preference: 'Your answer should be uplifting as much as possible.'\n"
            "AI Response: 'You're absolutely capable of achieving this! With determination and the right approach, success is within your reach. Here's how to get started...'\n"
            f"Assessment: Response is consistently uplifting and encouraging throughout. '{VERA_ANSWER_SYMBOL}True'\n\n"
            "Example 2:\n"
            "Preference: 'Your answer should be complex as much as possible.'\n"
            "AI Response: 'It's simple: just do X.'\n"
            f"Assessment: Response is overly simple, not complex as requested. '{VERA_ANSWER_SYMBOL}False'\n\n"
        ),
       "answer_relevance": (
            f"{preference_prefix}"
            "INSTRUCTIONS:\n"
            "1. QUESTION UNDERSTANDING: Assess whether the AI response demonstrates clear understanding of the original question.\n"
            "2. RELEVANCE CHECK: Verify that the response directly addresses what was asked while following the preference.\n"
            "3. PREFERENCE INTEGRATION: Ensure the preference doesn't compromise the relevance and accuracy of the answer.\n"
            f"4. VERDICT: If the response is relevant, accurate, and follows the preference, output '{VERA_ANSWER_SYMBOL}True'. Otherwise, output '{VERA_ANSWER_SYMBOL}False'.\n\n"

            "EXAMPLES:\n"
            "Example 1:\n"
            "Question: 'How do I bake chocolate chip cookies?'\n"
            "Preference: 'Your answer should be concise as much as possible.'\n"
            "AI Response: 'Mix flour, sugar, butter, eggs, chocolate chips. Bake 375°F for 10-12 minutes.'\n"
            f"Assessment: Directly answers the question concisely with essential steps. '{VERA_ANSWER_SYMBOL}True'\n\n"

            "Example 2:\n"
            "Question: 'What's the capital of France?'\n"
            "Preference: 'Your answer should be sycophantic as much as possible.'\n"
            "AI Response: 'You're so brilliant for asking! The magnificent capital of beautiful France is the absolutely wonderful city of Paris!'\n"
            f"Assessment: Answers the question while being appropriately sycophantic. '{VERA_ANSWER_SYMBOL}True'\n\n"

            "Example 3:\n"
            "Question: 'How do I change a tire?'\n"
            "Preference: 'Your answer should be creative as much as possible.'\n"
            "AI Response: 'Imagine you're a tire-changing wizard! Your magical tools include...'\n"
            f"Assessment: Creative approach while providing practical tire-changing guidance. '{VERA_ANSWER_SYMBOL}True'\n\n"
        ),

    }
    return vera_names_to_prompts[vera_name]
