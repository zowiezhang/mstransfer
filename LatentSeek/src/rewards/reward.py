import re
from termcolor import colored
from prompts.vera_prompts import get_vera_prompt
from prompts.vera_prompts import VERA_ANSWER_SYMBOL
# from prompts.auto_preference_generator import create_auto_preference_generator
import torch
import ipdb

class RewardModel(object):
    def __init__(
            self, 
            model,
            tokenizer,
            device: str = "cuda",
            # auto_preference: bool = True,
            # rule_format_string: str = None,
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
        # self.auto_preference = auto_preference
        # self.rule_format_string = rule_format_string
        
        # Initialize automatic preference generator
        # if self.auto_preference:
        #     self.preference_generator = create_auto_preference_generator() 



    def load_preference_alignment_verifiers(self):
        """
        Load verifiers for preference alignment evaluation.
        No longer domain-specific, works for any preference type.
        """
        veras = [
            "preference_alignment",
            # "style_consistency", 
            # "preference_completeness",
            # "answer_relevance",
        ]

        return veras
        

    def get_verifications(self, question: str, pref_input: str, solution: str):
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
        # if preference is None and self.auto_preference:
        #     preference = self.preference_generator.generate_preference_prompt(question)
        #     print(colored(f"Auto-generated preference: {preference}", "cyan"))
        
        for vera_type in veras:
            vera_prompt = get_vera_prompt(vera_type, question, pref_input, solution)
            message = [{"role": "user", "content": vera_prompt}]
            inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "\nassistant\n" in response:
                response = response.split("\nassistant\n", 1)[-1].strip()
            verifications[vera_type] = self.extract_verifier_approval(question + ' ' + pref_input, response)

        return verifications
    

    def extract_verifier_approval(self, pref_input, verifier_response):
        """
        According to the model's judgment of verifier_response, output an integer score between -5 and 0, indicating the degree to which the preference is satisfied.

        Args:
            verifier_response: str, the response from the verifier

        Returns:
            verifier_score: int, score in [-5, 0]
        """
        # Construct a scoring prompt to let the model directly give a score from -5 to 0
        judge_prompt = (
            "According to the following content, judge the degree to which the AI's answer satisfies the preference requirement. "
            "Only answer with a single integer score, ranging from -5 to 0. -5 means completely unsatisfied, 0 means fully satisfied.\n\n"
            f"Preference requirement: {pref_input}\n\n"
            f"Content:\n{verifier_response}\n\nYour score:"
        )
        message = [{"role": "user", "content": judge_prompt}]
        inputs = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # ipdb.set_trace()
        # Try to extract an integer between -5 and 0 from the model's response
        print(model_response + '\n\n')
        import re
        match = re.search(r"(-?\d+)", model_response[-30:])
        print(match.group(1) + '\n\n')
        # ipdb.set_trace()
        if match:
            score = int(match.group(1))
            if score < -5:
                score = -5
            elif score > 0:
                score = 0
            return score
        else:
            # If unable to parse, return 0 as neutral
            return -3

            
   
            

    def get_reward(self, question, pref_input, solution):
        '''
        Get preference alignment reward from question and solution.

        Args:
            question: str, the user's question/prompt
            pref_input: str, the preference input
            solution: str, the AI's response to be evaluated
        Returns:
            reward: float, normalized reward score between 0 and 1
        '''
        verifications = self.get_verifications(question, pref_input, solution)
        reward = 0
        reward_list = self.get_preference_reward_list()
        total_weight = sum([abs(w) for w in reward_list.values()])
        if total_weight == 0:
            normalized_weights = {k: 0 for k in reward_list}
        else:
            normalized_weights = {k: v / total_weight for k, v in reward_list.items()}
        
        # 使用加权求和方式计算reward
        # for verifier_name, verifier_score in verifications.items():
        #     weight = normalized_weights[verifier_name]
        #     reward += verifier_score * weight

        reward = verifications["preference_alignment"]
       
        print(colored(f"🏆 Final Preference Alignment Score: {reward:.3f}", "blue", attrs=["bold"]))
        
        return reward

        
    
    def get_preference_reward_list(self):
        '''
        Get reward weights for different preference alignment verifiers.
        Higher weights indicate more important aspects of alignment.
        '''
        reward_list = {
            "preference_alignment": 10,      # Most important: does it follow the preference?
            # "style_consistency": 0,        # Important: is the style consistent?
            # "preference_completeness": 0,  # Important: is the preference fully embodied?
            # "answer_relevance": 0,         # Most important: does it answer the question?
        }
        return reward_list
    
    
    

