# Text Generation with IMDB Reviews

## Project Overview
This assignment focuses on natural language generation techniques using Python. The project covers a range of text generation methods, including generating random text based on word frequencies, building Markov chains for sequence generation, and using GPT-2 for text completion tasks. We work with the IMDB movie reviews dataset to build frequency-based and Markov chain models, and then explore the pre-trained GPT-2 language model for creative text generation.

## Steps Completed

### Part 1: Loading and Decoding IMDB Reviews
**Objective:** To work with the IMDB reviews dataset, stored as tokenized word IDs.  
**Implementation:** Loaded the dataset in Colab and created a function to decode the reviews by mapping token IDs to words using a custom vocabulary.  
**Outcome:** Displayed a sample review in its decoded text format, facilitating readability for further tasks.

### Part 2: Frequency-Based Random Text Generation
**Objective:** To generate random text based on word frequencies in the IMDB reviews.  
**Implementation:**
- `add_to_dictionary`: Created a function to increment word counts in a dictionary.
- `create_frequency_dictionary`: Implemented a function to build a frequency dictionary for words across all reviews.
- `generate_random_text`: Generated random sequences of words by selecting words from the dictionary without weighting.
- `generate_random_weighted_text`: Generated more realistic random text by weighting word selection based on frequency.  
**Outcome:** Successfully generated both unweighted and weighted random sentences, with weighted text appearing more natural.

### Part 3: Markov Chain Text Generation
**Objective:** To generate text using a Markov chain based on word transitions in the reviews.  
**Implementation:**
- `create_markov_dictionary`: Built a Markov chain dictionary that maps each word to a list of possible subsequent words based on the reviews.
- `generate_markov_text`: Used the Markov chain dictionary to generate sequences of text by randomly selecting from the possible following words.  
**Outcome:** Generated plausible sentence-like sequences by chaining words according to observed transitions, adding coherence to the generated text.

### Part 4: Exploring GPT-2 for Advanced Text Generation
**Objective:** To generate and experiment with text completion using the GPT-2 model from HuggingFace.  
**Implementation:**
- **Loading GPT-2:** Used HuggingFace’s transformers library to load a pre-trained GPT-2 model and tokenizer.
- **Prompt-Based Generation:** Designed prompts to guide GPT-2 in generating text, experimenting with various topics and temperatures.
- **run_experiment:** Conducted experiments to explore GPT-2’s response to different prompts, such as varying genres and sentence structures.  
**Outcome:** Generated coherent, contextually relevant text based on the provided prompts, showcasing the advanced capabilities of GPT-2.

## Summary
Through this assignment, I implemented multiple techniques for text generation, from basic frequency-based methods to complex neural network-based generation with GPT-2. The project demonstrates the diversity of text generation methods, from simpler random sampling to using state-of-the-art language models. This exploration provided insights into how each method contributes differently to text generation, with GPT-2 showing particularly strong contextual and grammatical coherence.
