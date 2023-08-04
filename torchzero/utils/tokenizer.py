"""
Implements text tokenization schemes and algorithms. 

Author: Paul Wilson
"""
import os
import torch
import typing as tp 
from tqdm import tqdm


class Tokenizer:
    """
    Given a vocabulary of tokens, encodes or decodes target strings in and out from 
    sequences of integers. 
    """
    TOKEN_KEY = 0

    def __init__(self, vocab: list[str], use_unknown_token=True):
        assert len(set(vocab)) == len(vocab), 'There are duplicate words in the vocabulary!'

        self._token_graph = {}
        self.use_unknown_token = use_unknown_token
        self._unknown_token = 0
        self.vocab = []
        
        self.idx2vocab = {}
        self.idx2vocab[self._unknown_token] = '<UNK>'

        for i, word in enumerate(vocab): 
            self.add_token(word, i+1)
    
    @property
    def vocab2idx(self):
        return {
            v: k for k, v in self.idx2vocab.items()
        }

    def add_token(self, word, i=None): 
        i = i if i is not None else len(self.idx2vocab)
        if i in self.idx2vocab: 
            raise ValueError(f'Token {i} already represents the word {self.idx2vocab[i]}')
        else: 
            d = self._token_graph
            for letter in word: 
                d = d.setdefault(letter, {}) 
            d[self.TOKEN_KEY] = i 
        self.idx2vocab[i] = word
        self.vocab.append(word)

    def encode(self, text) -> list[int]: 
        #TODO this doesn't work. 

        tokens = []
        i = 0
        while i < len(text):  
            nodes = [self._token_graph]
            current_word = ""
            while i < len(text) and text[i] in nodes[-1]: 
                nodes.append(nodes[-1][text[i]])
                current_word += text[i]
                i += 1
            while 0 not in nodes[-1]:
                nodes.pop()
                current_word = current_word[:-1]
                i -= 1
            if current_word == "":
                if self.use_unknown_token:
                    tokens.append(self._unknown_token)
                    i += 1
                else:
                    raise ValueError(f"Found unknown token {current_word}!")
            else:
                tokens.append(nodes[-1][0])
        return tokens
    
    def decode(self, encoding): 
        return "".join([self.idx2vocab[i] for i in encoding])

    def __len__(self): 
        return len(self.idx2vocab)

class BytePairEncodingAlgorithm:
    tokens: set 
    idx2token: list[str]
    token2idx: dict 
    tokenfrequencies: dict
    _sorted_tokens_for_encoding: list = None

    UNKNOWN_TOKEN = -1

    def __init__(self, allow_whitespace: tp.Literal['prefix_only', True, False], max_tokens=1000):
        """
        Implements Byte Pair Encoding.
        Args: 
            allow_whitespace: whether to allow whitespace to be part 
                of bigram tokens. If set to True, allows all whitespaces to be fused.
                If false, allows no whitespace tokens to be fused. 
                If `prefix only`, allows whitespaces to only be fused as the prefix 
                to the bigram (used in GPT2)
            max_tokens (int): The maximum number of tokens this model will attempt to add to the vocabulary
        """
        self.allow_whitespace = allow_whitespace
        self.max_tokens = max_tokens

    def fit_v2(self, text): 
        # this implementation turned out to actually be slower than the first
        # raise DeprecationWarning
        # initial step
        max_tokens = self.max_tokens
        vocab = set(list(text))
        tokenizer = Tokenizer(vocab)
        
        bar = tqdm(desc='Computing Vocabulary', total=(max_tokens - len(tokenizer)))
        while len(tokenizer) <= max_tokens:
            encoding = tokenizer.encode(text)
            bigram_frequencies = self.bigram_frequencies(encoding)
            bigrams = list(bigram_frequencies.keys())
            bigrams.sort(key = lambda bigram: bigram_frequencies[bigram], reverse=True)
            
            most_frequent_bigram = None 
            for bigram in bigrams: 
                t1, t2 = bigram 
                if self._can_fuse_bigrams(tokenizer.idx2vocab[t1], tokenizer.idx2vocab[t2]): 
                    most_frequent_bigram = bigram 
                    break 
            if most_frequent_bigram is None: 
                from warnings import warn
                warn(f'BPE fitting finished without reaching max tokens because no remaining '
                     f'bigrams could be fused. ')
                return tokenizer
            else: 
                i1, i2 = most_frequent_bigram
                replacement_char = tokenizer.idx2vocab[i1] + tokenizer.idx2vocab[i2]
                tokenizer.add_token(replacement_char)

            bar.update(1)
            bar.set_postfix_str(f'added `{replacement_char}`')
        
        return tokenizer.vocab
    
    def fit(self, text): 
        """
        Uses the byte pair encoding algorithm to find a vocabulary for tokenizing the given 
        text. 
        """
        self.tokens = set()
        max_tokens = self.max_tokens

        _freqs = {} 
        for character in text: 
            self.tokens.add(character)
            _freqs.setdefault(character, 0)
            _freqs[character] += 1

        self.idx2token = list(self.tokens)
        self.token2idx = {character: idx for idx, character in enumerate(self.idx2token)}
        self.tokenfrequencies = {
            self.token2idx[char]: _freqs[char] for char in self.tokens
        }

        tokenization = [self.token2idx[char] for char in text]

        if max_tokens is None: 
            return 

        from tqdm.auto import tqdm
        bar = tqdm(desc='Computing Vocabulary', total=(max_tokens - len(self.tokens)))

        while len(self.tokens) < max_tokens:
            frequencies = self.bigram_frequencies(tokenization)
            bigrams = list(frequencies.keys())

            # sort list in descending order 
            bigrams.sort(key=lambda bigram: frequencies[bigram], reverse=True)
            
            # find first bigram that does not include whitespace
            most_frequent_bigram = None 
            for bigram in bigrams: 
                t1, t2 = bigram 
                if self._can_fuse_bigrams(self.idx2token[t1], self.idx2token[t2]): 
                    most_frequent_bigram = bigram 
                    break 

            if most_frequent_bigram is None: 
                raise ValueError('Fitting failed')

            i1, i2 = most_frequent_bigram
            replacement_char = self.idx2token[i1] + self.idx2token[i2]
            replacement_idx = len(self.tokens)

            self.tokens.add(replacement_char)
            self.idx2token.append(replacement_char)
            self.token2idx[replacement_idx] = replacement_char

            tokenization = self.replace_bigrams(tokenization, most_frequent_bigram, replacement_idx)

            bar.update(1)
            bar.set_postfix_str(f'added `{replacement_char}`')

        return list(self.tokens)

    def _can_fuse_bigrams(self, prefix, suffix): 
        # spaces should come before in tokenization
        if suffix.isspace(): 
            if self.allow_whitespace == 'prefix_only': return False
            else: return self.allow_whitespace
        # at most one space should be allowed to join
        if prefix.isspace(): 
            if self.allow_whitespace is True: return True
            if suffix[0].isspace(): return False
            elif prefix != ' ': return False 
            else: return False
    
        return True

    def _sort_key_for_encoding(self, token): 
        # primary key should be length, so that long tokens precede short ones. 
        # secondary key should be token frequency 
        return len(token), self.tokenfrequencies[self.token2idx[token]]

    def replace_bigrams(self, tokenization, bigram: tuple[int, int], replacement: int):
        out = []
        i = 0
        while i < len(tokenization):
            if i + 1 < len(tokenization):
                t1, t2 = tokenization[i:i+2]
                if (t1, t2) == bigram: 
                    out.append(replacement)
                    i += 2
                    self.tokenfrequencies[t1] -= 1
                    self.tokenfrequencies[t2] -= 1 
                    self.tokenfrequencies.setdefault(replacement, 0)
                    self.tokenfrequencies[replacement] += 1
                else:
                    out.append(tokenization[i])
                    i += 1
            else: 
                out.append(tokenization[i])
                i += 1
        return out   

    @staticmethod
    def bigram_frequencies(tokenization):
        frequencies = {}
        for i in range(len(tokenization) - 1): 
            bigram = tuple(tokenization[i:i+2])
            frequencies.setdefault(bigram, 0)
            frequencies[bigram] += 1 

        return frequencies

