import random
import numpy as np

class Clager(object):
    def __init__(self, dictionary, sent_ratio, word_ratio):
        self.dictionary = dictionary
        self.sent_ratio = sent_ratio
        self.word_ratio = word_ratio

    def clag(self, sent, cur_lang):
        if self.dictionary is None or cur_lang not in self.dictionary:
            return sent
        if random.random() >= self.sent_ratio:
            return sent
        words = []
        for word in sent.split(" "):
            if word not in self.dictionary[cur_lang]:
                words.append(word)
                continue
            elif random.random() >= self.word_ratio:
                words.append(word)
                continue
            else:
                lans = list(self.dictionary[cur_lang][word].keys())
                lan_max_id = len(lans) - 1
                lan_id = random.randint(0, lan_max_id)
                lan = lans[lan_id]
                target_words = self.dictionary[cur_lang][word][lan]
                word_max_id = len(target_words) - 1
                target_word_id = random.randint(0, word_max_id)
                target_word = self.dictionary[cur_lang][word][lan][target_word_id]
                words.append(target_word)
                continue
        return " ".join(words)

    def clag_word(self, word, cur_lang, is_pos=True):
        if self.dictionary is None or cur_lang not in self.dictionary:
            return word
        if is_pos:
            if word not in self.dictionary[cur_lang]:
                return word
        else:
            word_list = list(self.dictionary[cur_lang].keys())
            word_id = random.randint(0, len(word_list) - 1)
            word = word_list[word_id]
        lans = list(self.dictionary[cur_lang][word].keys())
        lan_max_id = len(lans) - 1
        lan_id = random.randint(0, lan_max_id)
        lan = lans[lan_id]
        target_words = self.dictionary[cur_lang][word][lan]
        word_max_id = len(target_words) - 1
        target_word_id = random.randint(0, word_max_id)
        any_word = self.dictionary[cur_lang][word][lan][target_word_id]
        return any_word

    def dclag(self, sent, cur_lang, num_pos_sample, num_neg_sample):
        if self.dictionary is None or cur_lang not in self.dictionary:
            return [sent] * (num_pos_sample + num_neg_sample)
        if random.random() >= self.sent_ratio:
            return [sent] * (num_pos_sample + num_neg_sample)
        words = list(enumerate(sent.split(" ")))
        random.shuffle(words)
        for idx, word in words:
            if word in self.dictionary[cur_lang]:
                break
        else:
            return [sent] * (num_pos_sample + num_neg_sample)
        lans = list(self.dictionary[cur_lang][word].keys())
        lan_max_id = len(lans) - 1
        lan_id = random.randint(0, lan_max_id)
        lan = lans[lan_id]
        dp = self.dictionary[cur_lang][word][lan]
        words = sent.split(" ")
        pos_samples = []
        for tgt in dp:
            pos_samples.append(list(words))
            pos_samples[-1][idx] = tgt
        neg_samples = []
        for _ in range(num_neg_sample):
            neg_samples.append(list(words))
            word_list = list(self.dictionary[cur_lang].keys())
            word_id = random.randint(0, len(word_list) - 1)
            word = word_list[word_id]
            lans = list(self.dictionary[cur_lang][word].keys())
            lan_max_id = len(lans) - 1
            lan_id = random.randint(0, lan_max_id)
            lan = lans[lan_id]
            target_words = self.dictionary[cur_lang][word][lan]
            word_max_id = len(target_words) - 1
            target_word_id = random.randint(0, word_max_id)
            any_word = self.dictionary[cur_lang][word][lan][target_word_id]
            neg_samples[-1][idx] = any_word
        pos_idx = np.random.choice(list(range(len(pos_samples))), size=num_pos_sample, replace=True)
        full_samples =  np.array(pos_samples)[pos_idx].tolist() + neg_samples
        return [" ".join(new_words) for new_words in full_samples]

        #sents = []
        #for _ in range(num_pos_sample):
        #    words = []
        #    for word in sent.split(" "):
        #        if word not in self.dictionary[cur_lang] or random.random() >= self.word_ratio:
        #            words.append(word)
        #            continue
        #        else:
        #            words.append(self.clag_word(word, cur_lang, is_pos=True))
        #    sents.append(" ".join(words))
        #for _ in range(num_neg_sample):
        #    words = []
        #    for word in sent.split(" "):
        #        if word not in self.dictionary[cur_lang] or random.random() >= self.word_ratio:
        #            words.append(word)
        #            continue
        #        else:
        #            words.append(self.clag_word(word, cur_lang, is_pos=False))
        #    sents.append(" ".join(words))
        ##print(sents)
        #return sents