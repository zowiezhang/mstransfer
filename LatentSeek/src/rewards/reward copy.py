import re
from termcolor import colored
from prompts.vera_prompts import get_vera_prompt
from prompts.vera_prompts import VERA_ANSWER_SYMBOL
from prompts.auto_preference_generator import create_auto_preference_generator
import torch

class RewardModel(object):
    def __init__(
            self, 
            model,
            tokenizer,
            device: str = "cuda",
            auto_preference: bool = True,
            rule_format_string: str = None,
        ):
        """
        Preference-based reward model for test-time alignment without dataset dependency.
        
        Args:
            model: the model to use for reward prediction
            tokenizer: the tokenizer to use for reward prediction
            device: device to run the model on
            auto_preference: whether to automatically detect preferences from user prompts
            rule_format_string: str, optional answer format constraints
        """

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.auto_preference = auto_preference
        self.rule_format_string = rule_format_string
        
        # Initialize automatic preference generator
        if self.auto_preference:
            self.preference_generator = create_auto_preference_generator() 



    def load_preference_alignment_verifiers(self):
        """
        Load verifiers for preference alignment evaluation.
        No longer domain-specific, works for any preference type.
        """
        veras = [
            "preference_alignment",
            "style_consistency", 
            "preference_completeness",
            "answer_relevance",
        ]

        return veras
        

    def get_verifications(self, question: str, solution: str, preference: str = None):
        '''
        Get preference alignment verifications from different verifiers.

        Args:
            question: str, the user's question/prompt
            solution: str, the AI's response to be evaluated
            preference: str, optional. If None, will auto-generate based on question

        Returns:
            verifications: dict, verifier_name -> verifier_approval
        '''
        veras = self.load_preference_alignment_verifiers()
        verifications = dict()
        
        # Auto-generate preference if not provided and auto_preference is enabled
        if preference is None and self.auto_preference:
            preference = self.preference_generator.generate_preference_prompt(question)
            print(colored(f"Auto-generated preference: {preference}", "cyan"))
        
        for vera_type in veras:
            vera_prompt = get_vera_prompt(vera_type, question, solution, preference)
            message = [{"role": "user", "content": vera_prompt}]
            inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            verifications[vera_type] = self.extract_verifier_approval(response)

        return verifications
    

    def extract_verifier_approval(self, verifier_response):
        '''
        Extract verifier approval from verifier response.

        Args:
            verifier_response: str, verifier response

        Returns:
            verifier_approval: bool, verifier approval
        '''
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        pattern = re.compile(
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)), 
            flags=re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(verifier_response)
        answer = match.group(1).strip() if match else None
        if not answer:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        f"and full verifier_response (length {len(verifier_response)}): "
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False
    
        answer = answer.replace("*", "")  # Remove any asterisks (bolding)
        answer = answer.strip().lower()

        if "true" in answer:
            return True
        elif "false" in answer:
            return False
        else:
            # Check if 'true' or 'false' is in the first word
            print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                        f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
            first_word = answer.split()[0]
            if "true" in first_word:
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                return True
            elif "false" in first_word:
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                return False
            else:
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                return False
            

    def get_preference_alignment_reward(self, question, solution, preference=None):
        '''
        Get preference alignment reward from question and solution.

        Args:
            question: str, the user's question/prompt
            solution: str, the AI's response to be evaluated
            preference: str, optional. If None, will auto-generate based on question
        Returns:
            reward: float, normalized reward score between 0 and 1
        '''
        verifications = self.get_verifications(question, solution, preference)
        reward = 0
        reward_list = self.get_preference_reward_list()
        total_weight = sum(reward_list.values())
        
        print(colored(f"\n🎯 Preference Alignment Evaluation:", "blue", attrs=["bold"]))
        for verifier_name, verifier_approval in verifications.items():
            weight = reward_list[verifier_name]
            if verifier_approval:
                print(colored(f"✅ {verifier_name}: APPROVED (weight: {weight})", "green"))
                reward += weight
            else:
                print(colored(f"❌ {verifier_name}: DISAPPROVED (weight: {weight})", "red"))

        # Optional format verification
        if self.rule_format_string is not None:
            format_approval = self.get_rule_format_verify(solution)
            if format_approval:
                print(colored(f"✅ Format verification: APPROVED", "green"))
            else:
                print(colored(f"❌ Format verification: DISAPPROVED", "red"))
                reward -= 0.1  # Small penalty for format issues
                
        # Normalize reward to [0, 1] range
        normalized_reward = max(0.0, reward / total_weight)
        print(colored(f"🏆 Final Preference Alignment Score: {normalized_reward:.3f}", "blue", attrs=["bold"]))
        
        return normalized_reward


    def get_rule_format_verify(self, solution):
        """
        Judge whether the answer follow the format rule.

        Args:
            solution: str
        """
        answer_pattern = self.rule_format_string
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        if len(matches) > 0:
            return True
        else:
            return False
        
    
    def get_preference_reward_list(self):
        '''
        Get reward weights for different preference alignment verifiers.
        Higher weights indicate more important aspects of alignment.
        '''
        reward_list = {
            "preference_alignment": 3,      # Most important: does it follow the preference?
            "style_consistency": 2,        # Important: is the style consistent?
            "preference_completeness": 2,  # Important: is the preference fully embodied?
            "answer_relevance": 3,         # Most important: does it answer the question?
        }
        return reward_list
    
    def get_preference_analysis(self, question, detailed=False):
        '''
        Get detailed preference analysis for a given question.
        
        Args:
            question: str, the user's question/prompt
            detailed: bool, whether to return detailed analysis
            
        Returns:
            dict with preference analysis results
        '''
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
    
    def evaluate_with_multiple_preferences(self, question, solution, top_k=2):
        '''
        Evaluate the solution against multiple potential preferences.
        Useful for comprehensive evaluation.
        
        Args:
            question: str, the user's question/prompt
            solution: str, the AI's response to be evaluated
            top_k: int, number of top preferences to evaluate against
            
        Returns:
            dict with results for each preference
        '''
        if not self.auto_preference:
            return {"error": "Auto preference is disabled"}
            
        preference_options = self.preference_generator.generate_multiple_preferences(question, top_k)
        results = {}
        
        print(colored(f"\n🔄 Multi-Preference Evaluation ({top_k} preferences):", "blue", attrs=["bold"]))
        
        for i, (pref_type, pref_prompt) in enumerate(preference_options):
            print(colored(f"\n--- Preference {i+1}: {pref_type} ---", "magenta"))
            reward = self.get_preference_alignment_reward(question, solution, pref_prompt)
            results[f"{pref_type}_{i+1}"] = {
                "preference_type": pref_type,
                "preference_prompt": pref_prompt,
                "alignment_score": reward
            }
            
        # Calculate overall score
        overall_score = sum(result["alignment_score"] for result in results.values()) / len(results)
        results["overall_score"] = overall_score
        
        print(colored(f"\n🎯 Overall Multi-Preference Score: {overall_score:.3f}", "blue", attrs=["bold"]))
        
        return results
    
    # Legacy compatibility methods
    def get_reward(self, question, solution):
        '''Legacy method for backward compatibility.'''
        print(colored("⚠️  Using legacy get_reward method. Consider using get_preference_alignment_reward instead.", "yellow"))
        return self.get_preference_alignment_reward(question, solution)
        
    def get_reward_list(self):
        '''Legacy method for backward compatibility.'''
        return self.get_preference_reward_list()


# Factory function and convenience aliases
def create_preference_reward_model(model, tokenizer, device="cuda", auto_preference=True):
    """
    Factory function to create a PreferenceAlignmentRewardModel.
    
    Args:
        model: the model to use for reward prediction
        tokenizer: the tokenizer to use for reward prediction  
        device: device to run the model on
        auto_preference: whether to automatically detect preferences
        
    Returns:
        PreferenceAlignmentRewardModel instance
    """
    return PreferenceAlignmentRewardModel(
        model=model,
        tokenizer=tokenizer, 
        device=device,
        auto_preference=auto_preference
    )


