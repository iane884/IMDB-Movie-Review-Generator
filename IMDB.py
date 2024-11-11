# -*- coding: utf-8 -*-

IN_COLAB = 'google.colab' in str(get_ipython()) if hasattr(__builtins__,'__IPYTHON__') else False

"""# Random text generation"""

# Importing libraries
import random
from pprint import pprint

# Load the IMDB reviews
if IN_COLAB:
  from keras.datasets import imdb
  TOP_WORD_COUNT = 8000
  (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=TOP_WORD_COUNT)
  print(train_data)

# Decoding reviews

def create_imdb_vocabulary():
  vocabulary = imdb.get_word_index()
  # The first indices are reserved
  vocabulary = {k:(v+3) for k,v in vocabulary.items()}
  vocabulary["<PAD>"] = 0
  # See how integer 1 appears first in the review above.
  vocabulary["<START>"] = 1
  vocabulary["<UNK>"] = 2  # unknown
  vocabulary["<UNUSED>"] = 3

  # reversing the vocabulary.
  # in the index, the key is an integer,
  # and the value is the corresponding word.
  token_to_word_dict = dict([(value, key) for (key, value) in vocabulary.items()])
  return token_to_word_dict

def decode_review(text):
    '''converts encoded text to human readable form.
    each integer in the text is looked up in the index, and
    replaced by the corresponding word.
    '''
    return ([token_to_word_dict.get(i, '?') for i in text])


token_to_word_dict = create_imdb_vocabulary()

# Test out that decode function
# Notice how the list of numbers turns into a list of words
# With a few "special" ones like "<START>" and "<UNK>" (unknown)
# In this kind of encoding, we can only have TOP_WORD_COUNT words,
# so words that are in our most-popular 8000 are replaced by "<UNK>",
test_review = train_data[0]
test_review_decoded = decode_review(test_review)
print("REVIEW TOKENS:", test_review)
print("REVIEW TEXT:  ", test_review_decoded)

def add_to_dictionary(dictionary, word):

  # Task 0
  # Implement "add_to_dictionary"
  # Add a single word to the dictionary
  # Add one to its entry if it is in the dictionary
  # Otherwise set the value to 1

  return None

if IN_COLAB:
  test_dict = {}
  add_to_dictionary(test_dict, "cat")
  add_to_dictionary(test_dict, "cat")
  add_to_dictionary(test_dict, "cat")
  add_to_dictionary(test_dict, "mouse")
  print(test_dict)
  assert test_dict["cat"] == 3

def create_frequency_dictionary(samples):
  '''
  Parameters: samples: ndarray of reviews,
    each review is a list of ints, representing the tokenized words in that review
  Returns: dict of word-> word count
  '''

  # Task 1
  # Implement "create_frequency_dictionary"
  # For each sample in a list of samples, decode it using decode_review
  # and add each token to the dictionary
  # Note that this time, we don't need to tokenize it ourselves,
  # it has already been converted to tokens in the dataset
  # We won't worry about positive and negative dictionaries here
  # We are making a frequency dictionary for *all* reviews
  frequency = {}
  for review in samples:
    pass
  return frequency

if IN_COLAB:
  # Make a dictionary of the first four reviews
  # (they are long reviews, we end up with quite a few words!)
  test_freq_dict = create_frequency_dictionary(train_data[0:4])
  print(test_freq_dict)
  assert test_freq_dict["brilliant"] == 3

def generate_random_text(freq_dict, count=10):
  '''
  Parameters:
    freq_dict: dict of word-> word count
    count: how many words to generate
  Returns:
    str: randomly selected words, joined with " "
  '''

  # Task 2
  # Generate some random text from this dictionary
  # * Get the list of all keys (words) from this dictionary
  #     This returns a not-quite-list, so cast it to a list
  # * Use random.choice to add random words to a list until you have "count" number of words
  # * return that list, joined with " " so that you return a single string

  words = []

  return " ".join(words)

if IN_COLAB:
  for i in range(0,5):
    random_text = generate_random_text(test_freq_dict, 11)
    print(f"Randomly generated: '{random_text}'")

  # Notice: these sentences aren't grammatical,
  # but also they have a lot of uncommon words

  assert isinstance(random_text, str), "Make sure you are returning a string"
  assert len(random_text.split(" ")) == 11, "Make sure you are making the right number of words, joined with a space"

def generate_random_weighted_text(freq_dict, count=10):
  '''
  Parameters:
    freq_dict: dict of word-> word count
    count: how many words to generate
  Returns:
    str: randomly selected words, joined with " "
  '''

  # Task 3
  # The previous generator gave each word the same probability to get picked
  # We can get more natural-looking results with a *weighted* distribution
  # Use random.choices to get a list of "count" words,
  #  and return them joined with " "
  # (note that unlike "choice" it gives you a list so you can get many words in one function call)
  # In addition to your list of options (the keys) you will also need a list of "weights"
  # You can use dict.values() to get those!

  words = []

  return " ".join(words)

if IN_COLAB:
  for i in range(0,5):
    random_text = generate_random_weighted_text(test_freq_dict, 11)
    print("Randomly generated:", random_text)

  # Notice that this starts looking more like normal sentences,
  # though it still doesn't make sense, the words don't *go together*
  assert isinstance(random_text, str), "Make sure you are returning a string"
  assert len(random_text.split(" ")) == 11, "Make sure you are making the right number of words, joined with a space"

# Markov chains
# A Markov chain is a way to generate text by building a lookup table
# of *what words* can follow *what other words*


def create_markov_dictionary(samples):
  """
  Make a Markov dictionary of which words follow which other words
  Parameters: samples: ndarray of reviews,
    each review is a list of ints, representing the tokenized words in that review
  Returns: dict of str->list of str
  """
  # Task 4
  # Implement "create_markov_dictionary"
  # For each sample in a list of samples, decode it into words using decode_review
  # * Add this sample to the markov dictionary:
  #   * Create a last_word variable, and set it to words[0].
  #       This is our starting word (it will always be "<START>" in this data set)
  #   * For each word after this (suggested: use a slice, eg [1:])
  #     * We want to add this to the list of words-that-come-after last_word
  #         e.g. for text "...this story"
  #             last_word="this", markov = {..."this": [..."cat", "castle", "story"], ....}
  #     * Check to see if last_word has an entry in the markov dictionary yet
  #     * If it doesn't, make an entry of an empty list
  #     * Add the current word to the last_word's list, only if it doesn't appear in that list yet
  #         If we didn't do this, we would end up with *weighted* random choices,
  #             ...which might be good too!
  #   * Set the last_word to the current word.  Now *it* is the last word
  # You will end up with a dictionary of words that can follow each word
  # e.g. {...'were': ['great', 'just', 'watching', 'children'],....}


  markov = {}
  for review in samples:
    words = decode_review(s)

    # We will add "<END>" to the list so we know where to stop later
    words.append("<END>")
    # ....

  return markov

if IN_COLAB:
  # Make a markov dictioanry of the first four reviews
  test_markov_dict = create_markov_dictionary(train_data[0:4])
  # pprint(test_dict) # Print the dictionary more readably, but takes more space
  print("Markov dictionary", test_markov_dict)
  after_you = test_markov_dict["you"]
  after_just = test_markov_dict["just"]
  print("you->", after_you)
  print("just->", after_just)
  assert after_just == ['brilliant', 'imagine', 'so', '<UNK>', 'sat', 'started'], "Make sure you don't add duplicates"

# Generate some text with Markov chains

def generate_markov_text(markov_dict, starting_words, count):
  """
  Use this markov dictionary to generate text
  Parameters:
    markov_dict: dict of str->list of str
    current_words: list of str
    count: maximum number of words to generate (we might stop sooner when we hit <END>)
  Returns:
    str: the original words and all newly generated words joined by a " "
  """

  # Task 5
  # Use a markov dictionary to generate text
  # Make a variable to track the current word.
  # Set it to the *last* word of current_words to start with
  # (that's the word we will be continuing at)
  # While there are fewer than *count* words
  #   AND the current word is not "<END>"
  #   AND there is an entry for this word...
  #   .. get the options for words that can follow this word from the markov_dict
  #   .. pick a random one
  #   .. add that word to the list of current words
  # Once one of those conditions is true (ie, the while-loop terminates)
  # return all the words joined by " "

  # Make a copy of the starting words that we can add to
  current_words = starting_words[:]

  return " ".join(current_words)


if IN_COLAB:
  test_text = generate_markov_text(test_markov_dict, ["the", "witty"], 5)
  test_end_text = generate_markov_text(test_markov_dict, ["experienced"], 5)
  for (key, val) in test_markov_dict.items():
    if "<END>" in val:
      print(key, val)
  print("Test markov: ", test_text)
  print("Test markov end-early text: ", test_end_text)
  assert isinstance(test_text, str), "Should return a string"
  assert test_end_text == "experienced <END>", "Make sure you end when there is an '<END>' symbol"
  assert test_text.startswith("the witty remarks"), "'remarks' is the only option after 'witty', make sure you are continuing after the *last* word"
  assert len(test_text.split(" "))==5

   # We will "train" this markov dictionary on more reviews to give us more options
  large_markov_dict = create_markov_dictionary(train_data[0:400])

  for i in range(0, 5):
    markov_text = generate_markov_text(large_markov_dict, ["<START>"], 11)
    print(f"Markov text:  '{markov_text}'")

  print(f"Random text:  '{generate_random_text(test_freq_dict, 11)}'")
  print(f"Weighted text: '{generate_random_weighted_text(test_freq_dict, 11)}'")


  # Markov text can also generate text until it reaches an <END> word or
  #  runs into a word with no options. Lets try to generate a million words,
  #  and see how far it gets
  long_markov = generate_markov_text(large_markov_dict, ["<START>"], 1000000)
  print(f"Long markov:, len {len(long_markov)} words  '{long_markov}'")


  # We can also "steer" it by starting with a word we want
  # for i in range(0,5):
  #   western_markov = generate_markov_text(large_markov_dict, ["i", "like", "western"], 12)
  #   print(f"Western text:  '{western_markov}'")
  # for i in range(0,5):
  #   romance_markov = generate_markov_text(large_markov_dict, ["i", "like", "romantic"], 12)
  #   print(f"Romantic text: '{romance_markov}'")

"""# HuggingFace GPT2"""

# Using a neural network
!pip install transformers

if IN_COLAB:
  from transformers import GPT2Tokenizer
  from transformers import TFGPT2LMHeadModel

if IN_COLAB:
  # Make the tokenizer and the model, and load their weights from Huggingface
  # (can take a while)
  gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  gpt2_model = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=gpt2_tokenizer.eos_token_id)

if IN_COLAB:
  # We saw one kind of *encoded* versions of text in the movie reviews above
  # There, each word had a numerical "token" counterpart. One word, one token.
  # But we only had a limited number of tokens, so there were a lot of <UNK> unknown words
  test_review = train_data[0]
  test_review_decoded = decode_review(test_review)
  print("TOKENS", test_review)
  print("DECODED", test_review_decoded)

if IN_COLAB:
  # But we often want to tokenize new text, and more efficiently than one-word-one-token
  # So GPT2 uses a specially-trained tokenizer that can squish *any* word into smaller encodings

  gpt2_tokenizer("I love lamps")['input_ids']
  things = ["coffee", "COFFEE", "cats", "german shepherds", "zoodles", "blorpsquizzles", "computer science", "doing computer science homework", "CS150"]

  # Lets try encoding these statements
  for thing in things:
    text = f"I love {thing}"
    tokens = gpt2_tokenizer.encode(text)

    print(text, tokens)

if IN_COLAB:
  # We can also *decode* the text

  for thing in things:
    text = f"I love {thing}"
    tokens = gpt2_tokenizer.encode(text)
    # What if we change a token and try to decode?
    tokens[2] += 1
    decoded_text = gpt2_tokenizer.decode(tokens)
    print(text, tokens)
    print(" -> ", decoded_text)

if IN_COLAB:
  # Let's try generating some prompts


  prompt = 'The most fun thing to make in Python is '
  # prompt = 'I was late for magic school. Professor Dragon said'
  # prompt = 'The most unusual Starbucks drink contains'

  def generate_gpt2_output(prompt, max_tokens_to_generate=5, count=1, temperature=0.7):
    # Turn the prompt into tokens
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='tf')

    # Make this many more tokens
    max_length = input_ids.shape[1] + max_tokens_to_generate


    # generate text until the output length (which includes the prompt length) reaches 50
    # Temperature is how "weird" or "normal" the prompt is
    # top_k and top_p are other tuning parameters
    all_outputs = gpt2_model.generate(
      input_ids,
      do_sample=True,
      max_length=max_length,
      top_k=50,
      top_p=0.95,
      temperature=temperature,
      num_return_sequences=count)

    outputs = [gpt2_tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in all_outputs]
    return outputs

  outputs = generate_gpt2_output(prompt, count=5, max_tokens_to_generate=25)
  for output in outputs:
    print("--------", "\n", output)

def run_experiment(prompt, options):
  """
  Runs a GPT2 experiment by generating from several similar prompts
  Parameters:
    prompt(str) a prompt in the form "[X] is my favorite", "[X] was president of"
    options(list of str): options for the experiment, to replace the [X]
  Returns
    dict of list of str: the text generated for each option (without the original prompt)
  """
  # Task 6:
  # Run an experiment by replacing all instances of [X] with each one of these options

  # For each option
  #   create a modified version of the prompts with all the [X] replace with that option
  #
  #   Use "generate_gpt2_output" to generate 5 outputs, with 10 tokens, for each option
  #   This will return a list of outputs.
  #      Use a list comprehension and slicing to turn that into a list of just the generated parts
  #       e.g. modifying the prompt  "[X] is my favorite" => "Pizza is my favorite"
  #       e.g modifying the output ["Pizza is my favorite food, I love ..."] => [" food, I love ..."]
  #   Store the result at the key of that option, e.g. {"Pizza": [" food, I love ..."]}
  # Try adjusting the temperature to see if you get better results 

  results = {}
  for option in options:
    modified_prompt = prompt
    outputs = []
    # Uncomment to see the output as it is generated
    # for output in outputs:
    #   print("--------", "\n", output)

  return results

if IN_COLAB:
  # results = run_experiment("the best [X] flavor is", ["pizza", "coffee", "candy"])
  # results = run_experiment("the weather in [X] is", ["Chicago", "New York", "Miami", "Bangalore", "Sydney"])
  results = run_experiment("This [X] movie is", ["romantic", "western", "scifi", "dramatic", "terrible"])

  # This is the kind of experiment that can be done to detect "fairness" in a generator
  # results = run_experiment("[X] works at the hospital, [X] is a", ["she", "he"])

  pprint(results)

  assert isinstance(results, dict), "Should be a dictionary"
  # Your results will vary, but you should get results like this, depending on the experiment:
  """
   'romantic': [' a very good one.\nThe movie is about',
              ' a great way to get a little involved in your',
              ' a wonderful film, but it is not a movie',
              " a wonderful film, and it's a perfect example",
              ' a classic of the genre, with a very high'],
 'scifi': [' a great way to get started with the idea of',
           ' a great movie, but I’m not',
           ' an homage to the classic sci-fi novel by',
           ' a great example of how to make a movie that',
           ' a great movie and a great movie.\n\n'],
  """
