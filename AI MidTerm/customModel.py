import torch
import math
from nltk import tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CustomModelInference:

    def __init__(self):
        """
        The __init__ function sets up the device for computation (GPU if available),
        initializes a tokenizer and a GPT-2 language model, and assigns them to instance variables within the CustomModelInference class
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("GPU Activated!")
            torch.cuda.set_device(0)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.to(self.device)

    def calculate_log_likelihood(self, target_sentence, topic, previous_sentences, model="gpt2"):
        """
        The function takes a target sentence, a topic, and potentially previous sentences, tokenizes and encodes them,
        feeds them to a language model (GPT-2), computes token likelihoods, and then aggregates these likelihoods
        to calculate the log-likelihood of the target sentence.
        :param target_sentence: Target sentence to calculate log-probs
        :param topic: Topic of the story (the summary is used)
        :param previous_sentences: Previous history if none then just the topic
        :param model: GPT-2
        :return:
        """
        prompt = topic
        if previous_sentences and len(previous_sentences) > 0:
            prompt += " " + " ".join(previous_sentences)
        likelihood = 1.0
        log_likelihood = 0
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        prompt_tokens_length = len(prompt_tokens[0])
        target_sentence_tokens = self.tokenizer.encode(target_sentence, return_tensors="pt").to(self.device)

        combined_token = torch.cat((prompt_tokens, target_sentence_tokens), 1)

        output = self.model(combined_token)
        logits = output[0]
        with torch.no_grad():
            for i in range(prompt_tokens_length, len(combined_token[0])):
                next_token_prob = torch.softmax(logits[:, i - 1, :], dim=-1)
                next_token_index = combined_token[0][i]
                token_likelihood = next_token_prob[0][next_token_index].item()
                likelihood *= token_likelihood
                log_likelihood += math.log(token_likelihood, 2)

        return log_likelihood

    def process_sentences(self, sentences, topic, history_size=math.inf, model="gpt2"):
        """
        The function processes a text by tokenizing it into sentences, calculates the log-likelihoods of each sentence with respect to the topic,
        considers contextual information, computes c-values for each sentence, and returns the collective c-value along
        with individual topic-driven and contextual log-likelihoods for each sentence
        :param sentences: Sentence string
        :param topic: Topic of the story (the summary is used)
        :param history_size: History size
        :param model: GPT-2
        :return:
        """
        model = "gpt2"
        if not history_size:
            history_size = math.inf
        sentence_tokens = tokenize.sent_tokenize(sentences)
        topic_driven_output = []
        contextual_output = []

        for i in range(len(sentence_tokens)):
            print(f"Processing sentence {i + 1} out of {len(sentence_tokens)}")
            topic_driven_output.append(self.calculate_log_likelihood(sentence_tokens[i], topic, None, model))
            history_end = i
            history_start = max(0, i - history_size)
            contextual_output.append(self.calculate_log_likelihood(sentence_tokens[i], topic,
                                                                    sentence_tokens[history_start:history_end], model))
        c_value = 0

        for i in range(len(sentence_tokens)):
            c_value += (topic_driven_output[i] - contextual_output[i]) / (-1 * len(sentence_tokens[i]))

        return (c_value / len(sentence_tokens), topic_driven_output, contextual_output)

if __name__ == "__main__":
    custom_model = CustomModelInference()
