import torch

from torch.nn.functional import log_softmax, softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyApproach:
    """
    Class to handle the approach for the model.
    """

    def __init__(self, model_name, device="cuda"):
        """
        Initialize the Approach class with the model name and device.

        Args:
            model_name (str): The name of the model to load.
            device (str): The device to load the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def prepare_msgs(self, query, context):
        """
        Prepare the system and the user messages for the given query and context.

        Args:
            query (str): The user query.
            context (str): The system context.
        """
        # Create a list of messages starting with the system message.
        messages = [{"role": "system", "content": context}]
        messages.append({"role": "user", "content": query})
        return messages
    

    def generate_response(self, context, query, verbose=True):
        """
        Generate a response from the model based on the instruction.

        Args:
            context (str): The system context.
            query (str): The user query including the instruction passed to the model.
            verbose (bool): Whether to print the generated response.
        """
        
        if verbose:
            print("==="*7 + "CONTEXT:" + "==="*7)
            print(context)
            print()
            print("==="*7 + "QUERY:" + "==="*7)
            print(query)
            print()
        
        messages = self.prepare_msgs(query, context)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            # num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            # output_scores=True,
            # return_dict_in_generate=True
            # temperature=0.2,
            # top_p=0.9,
            # do_sample=True,
            # pad_token_id=tokenizer.eos_token_id
        )

        input_length = inputs["input_ids"].shape[-1]
        result = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        # ["sequences"]
        # print(self.tokenizer.decode(outputs["scores"], skip_special_tokens=True))

        if verbose:
            print("==="*7 + "RESULT:" + "==="*7)
            print(result)

        return result

    def get_token_logprobs(self, context, query, allowed_tokens=["A", "B", "C", "D"], temperature=1.0, verbose=True):
        """
        Get the log probabilities for specific allowed tokens for the next generated token.
        This simulates multiple choice question answering (MCQA).

        Args:
            context (str): The system context.
            query (str): The user query, ending with a prompt for a single letter.
            allowed_tokens (list): A list of allowed tokens to score (e.g. ["A", "B", "C", "D"]).
            temperature (float): Temperature for scaling the logits.
            verbose (bool): Whether to print debug information.
            
        Returns:
            dict: A dictionary mapping allowed tokens to their computed probabilities.
        """
        if verbose:
            print("==="*7 + "CONTEXT:" + "==="*7)
            print(context)
            print()
            print("==="*7 + "QUERY:" + "==="*7)
            print(query)
            print()

        messages = self.prepare_msgs(query, context)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)

        # We only need the logits for the very next token
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True
            )

        # The logits for the next token to be generated
        next_token_logits = outputs.logits[0, -1, :]

        # Extract logits for the allowed tokens
        token_logits = {}

        # The tokenizer might handle single letters with or without spaces weirdly. 
        # For letters like ' A', ' B', ' C', ' D' usually used after a newline, the token might be just the letter
        # Or it could be a specialized token. We try to be very specific by taking the first token ID of the string.
        for token in allowed_tokens:
            # specifically for Qwen, letters like "A", "B" etc are mapped directly.
            token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
            logit_val = next_token_logits[token_id].item()
            token_logits[token] = logit_val

        if verbose:
            print("Raw Logits:", token_logits)

        # Apply temperature scaling
        logits_tensor = torch.tensor([token_logits[t] for t in allowed_tokens], dtype=torch.float32)
        
        # Scaling
        scaled_logits = logits_tensor / temperature
        
        # Softmax to get probabilities
        probs = softmax(scaled_logits, dim=0).tolist()
        
        result_probs = {token: prob for token, prob in zip(allowed_tokens, probs)}
        
        if verbose:
            print("Probabilities:", result_probs)
            
        return result_probs