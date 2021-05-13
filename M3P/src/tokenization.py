import sentencepiece as spm
import collections

SPIECE_UNDERLINE = "▁"

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class XLMRTokenizer(object):
    def __init__(self, vocab_file,special_token=''):
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(str(vocab_file))
        self.sp_model = sp_model
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.sep_token = "</s>"
        self.cls_token = "<s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"

        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.fairseq_tokens_to_ids["<mask>"] = len(sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        self.cls_token_id = self._cls_token_id()
        self.sep_token_id = self._sep_token_id()
        self.pad_token_id = self._pad_token_id()
        self.mask_token_id = self._mask_token_id()
        self.eos_token_id = self._eos_token_id()


    def _cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self._convert_token_to_id(self.cls_token)
    def _sep_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self._convert_token_to_id(self.sep_token)
    def _pad_token_id(self):
        return self._convert_token_to_id(self.pad_token)
    def _eos_token_id(self):
        return self._convert_token_to_id(self.eos_token)
    def _mask_token_id(self):
        return self._convert_token_to_id(self.mask_token)

    def build_inputs_with_special_tokens(
            self,token_ids_0, token_ids_1):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        return vocab

    def _tokenize(self,text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self,token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.fairseq_tokens_to_ids[self.unk_token]

    def encode(self,text,text_b=None):
        tokens = self._tokenize(text)
        input_ids = []
        for token in tokens:
            input_ids.append(self._convert_token_to_id(token))

        input_ids_b = None
        if text_b is not None:
            input_ids_b = []
            tokens_b = self._tokenize(text_b)
            for token in tokens_b:
                input_ids_b.append(self._convert_token_to_id(token))

        #input_ids = self.build_inputs_with_special_tokens(input_ids,input_ids_b)

        return input_ids

    def decode(self,token_ids):
        _out = []
        for token_id in token_ids:
            _out.append(self._convert_id_to_token(token_id))
        return self.convert_tokens_to_string(_out)

    def _convert_id_to_token(self,index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

