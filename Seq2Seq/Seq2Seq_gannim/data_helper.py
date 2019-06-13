#-*- coding: utf-8 -*-

import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class dataHelper(object):
    # S: 디코딩 입력의 시작을 나타내는 심볼
    START_SYMBOL = ''
    # E: 디코딩 출력을 끝을 나타내는 심볼
    PAD_SYMBOL = ''
    # P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
    END_SYMBOL = ''

    def __init__(self, filename):
        self.load_source_target(filename)

    def get_word_dic(self, tot_datas):
        inputs = []
        outputs = []
        targets = []
        tot_word_idx_dic = {self.PAD_SYMBOL:0, self.START_SYMBOL:1, self.END_SYMBOL:2} # 
        for input, output in tot_datas:
            inputs.append(self.cut_utf8(input))
            # 디코더 셀의 입력값. 시작을 나타내는 S() 심볼을 맨 앞에 붙여준다.
            outputs.append(self.cut_utf8('{}{}'.format(self.START_SYMBOL,output))) 
            # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
            targets.append(self.cut_utf8('{}{}'.format(output, self.END_SYMBOL))) 
            self.update_word_idx_dic(tot_word_idx_dic, inputs[-1])
            self.update_word_idx_dic(tot_word_idx_dic, outputs[-1])
            self.update_word_idx_dic(tot_word_idx_dic, targets[-1])
        return inputs, outputs, targets, tot_word_idx_dic

    def get_word_idx(self, datas):
        inputs = [] 
        outputs = []
        for x, y in datas:
            inputs.append(self.cut_utf8(x))
            outputs.append(self.cut_utf8('{}{}'.format(self.START_SYMBOL,y)))
        input_inputs = self.convert_word_to_idx(inputs, self.tot_word_idx_dic)
        input_outputs = self.convert_word_to_idx(outputs, self.tot_word_idx_dic)
        return input_inputs, input_outputs, max([len(txt) for txt in inputs]), max([len(txt) for txt in outputs])
    
    def get_input_idxs(self, word):
        terms = self.cut_utf8(word)
        return np.array([np.array([self.tot_word_idx_dic[word] for word in terms])])
    
    def load_source_target(self, filename):
        # form. (source, target)
        self.tot_datas = self.read_data(filename) 
        # origin source, target, word to idx dicts ..  
        self.inputs, self.outputs, self.targets, self.tot_word_idx_dic = self.get_word_dic(self.tot_datas)
        ##
        self.max_inputs_seq_length = max([len(txt) for txt in self.inputs])
        self.max_outputs_seq_length = max([len(txt) for txt in self.outputs])
        self.max_targets_seq_length = max([len(txt) for txt in self.targets])
        self.max_sequence_length = max([self.max_inputs_seq_length, self.max_outputs_seq_length, self.max_targets_seq_length])
        ## idx to word
        self.tot_dic_len = len(self.tot_word_idx_dic)
        self.tot_idx_word_dic = self.get_idx_word_dic(self.tot_word_idx_dic)
        ## convert to array
        self.input_inputs = self.convert_word_to_idx(self.inputs, self.tot_word_idx_dic)
        self.input_outputs = self.convert_word_to_idx(self.outputs, self.tot_word_idx_dic)
        self.input_targets = self.convert_word_to_idx(self.targets, self.tot_word_idx_dic)

    def get_translated_str(self, result):
        """ decoder 결과를 string 으로 변환 ( idxs -> string )"""
        decoded_str = ''.join([self.tot_idx_word_dic[n] for n in result])
        if self.END_SYMBOL in decoded_str:
            translated = ''.join(decoded_str[:decoded_str.index(self.END_SYMBOL)])
        else:
            translated = decoded_str
        return translated
        
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """ batch iterator 생성 및 전달 """
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                suffled_data = data[shuffle_indices]
            else:
                suffled_data = data
            for bnum in range(num_batches_per_epoch):
                sidx = bnum * batch_size
                eidx = min((bnum+1)*batch_size, data_size)
                yield suffled_data[sidx:eidx]

    def get_suffled_data(self):
        """ suffling data """
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(self.targets)))

        inputs_suffled = self.input_inputs[shuffle_indices]
        outputs_suffled = self.input_outputs[shuffle_indices]
        targets_suffled = self.input_targets[shuffle_indices]

        ## split train/test set
        dev_idx = -1 * int(0.1 * float(len(self.targets)))
        inputs_train, inputs_dev = inputs_suffled[:dev_idx], inputs_suffled[dev_idx:]
        outputs_train, outputs_dev = outputs_suffled[:dev_idx], outputs_suffled[dev_idx:]
        targets_train, targets_dev = targets_suffled[:dev_idx], targets_suffled[dev_idx:]
        return inputs_train, inputs_dev, outputs_train, outputs_dev, targets_train, targets_dev
        
    @staticmethod 
    def pad(x, batch_size, max_length):
        pad_x = np.zeros((batch_size, max_length))
        x_seq_len = np.zeros((batch_size))
        x_size = len(x)
        for idx in range(batch_size):
            if x_size <= idx:
                x_seq_len[idx] = len(x[x_size-1])
                pad_x[idx][:int(x_seq_len[idx])] = x[x_size-1]
            else:
                x_seq_len[idx] = len(x[idx])
                pad_x[idx][:int(x_seq_len[idx])] = x[idx]
        return pad_x, x_seq_len

    @staticmethod
    def cut_utf8(str_input):
        str_input = unicode(str_input) #if type(str_input) is str else str_inputs
        chars = list(str_input.lower())
        chars = [char.encode('utf-8') for char in chars]
        return chars
    
    @staticmethod
    def update_word_idx_dic(word_idx_dic, words):
        for word in words:
            if word not in word_idx_dic:
                word_idx_dic[word] = len(word_idx_dic)
        return word_idx_dic
    
    @staticmethod
    def get_idx_word_dic(word_idx_dic):
        return {word_idx_dic[key]:key for key in word_idx_dic.keys()}
            
    @staticmethod
    def convert_word_to_idx(datas, word_idx_dic):
        return np.array([np.array([word_idx_dic[word] for word in data]) for data in datas])
    
    @staticmethod
    def read_data(filename):
        with open(filename, 'r') as ff:
            return [line.strip().split('\t') for line in ff.readlines()]
    
    @staticmethod
    def save_vocab(vocab_path, tot_dic_len, tot_dic):
        with open('{}'.format(vocab_path), 'w') as ff:
            ff.write("tot_dic_len\n")
            ff.write("{}\n".format(tot_dic_len))
            ff.write("tot_word_idx_dic\n")
            for key in tot_dic:
                ff.write("{}\t{}\n".format(key, tot_dic[key]))
            ff.write("\n")

