import random
import json
import numpy as np
random.seed(1234)
max_l = 32 # max length of sentences

def process_sequence(cell, inputs, masks):
    '''
    Processes a whole sequence with an RNN cell and returns its last hidden state.

    Parameters
    ----------
    cell: RNN Cell
    inputs: float32 Tensor of dimensions [batch_size, max_l, 300]
    masks: int Tensor of dimensions [batch_size, max_l, 1]

    Returns
    -------
    state: Float32 tensor of dimensions [batch_size, hidden_states]
    '''
    state = cell.zero_state(inputs)
    for i in range(max_l):
        state = cell(inputs[:, i], state)*masks[:, i] + (1.-masks[:, i])*state
    return state

class BatchToken:
    def __init__(self):
        self.__count = 0
    def incr(self):
        self.__count += 1
    def reset(self):
        self.__count = 0
    def val(self):
        return self.__count

class Dataset:
    def __init__(self, train, val, test, word2vec):
        self.train = train
        self.val = val
        self.test = test
        self.word2vec = word2vec
        self.token = BatchToken()

    def next_batch(self, batch_size):
        '''
        Returns next batch of training data of size batch_size (and shuffles the dataset at the beginning of every epoch)

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        batch_x: float32 numpy array of dimensions [batch_size, max_l, 300]
        batch_m: int numpy array of dimensions [batch_size, max_l, 1]
        batch_y: int numpy array of dimensions [batch_size, 1]
        '''
        token = self.token
        train = self.train
        if ((token.val()+1)*batch_size+1) > len(train):
            random.shuffle(train)
            token.reset()
        start = token.val()*batch_size
        end = (token.val()+1)*batch_size
        res = train[start:end]
        batch_y = [[x[1]] for x in res]
        batch_x = [x[0] for x in res]
        batch_x = [[self.get_w2v(y) for y in x] for x in batch_x]
        batch_x, batch_m = np.array([[np.zeros(300) for i in range(max_l - len(x[:max_l]))] + x[:max_l] for x in batch_x]), np.array([[[0.] for i in range(max_l - len(x))] + [[1.] for i in range(len(x[:max_l]))] for x in batch_x])
        token.incr()
        return batch_x, batch_m, batch_y

    def test_batch(self):
        '''
        Returns test data

        Returns
        -------
        batch_x: float32 numpy array of dimensions [batch_size, max_l, 300]
        batch_m: int numpy array of dimensions [batch_size, max_l, 1]
        batch_y: int numpy array of dimensions [batch_size, 1]
        '''
        res = self.test
        batch_y = [[x[1]] for x in res]
        batch_x = [x[0] for x in res]
        batch_x = [[self.get_w2v(y) for y in x] for x in batch_x]
        batch_x, batch_m = np.array([[np.zeros(300) for i in range(max_l - len(x[:max_l]))] + x[:max_l] for x in batch_x]), np.array([[[0.] for i in range(max_l - len(x))] + [[1.] for i in range(len(x[:max_l]))] for x in batch_x])
        return batch_x, batch_m, batch_y

    def val_batch(self):
        '''
        Returns validation data

        Returns
        -------
        batch_x: float32 numpy array of dimensions [batch_size, max_l, 300]
        batch_m: int numpy array of dimensions [batch_size, max_l, 1]
        batch_y: int numpy array of dimensions [batch_size, 1]
        '''
        res = self.val
        batch_y = [[x[1]] for x in res]
        batch_x = [x[0] for x in res]
        batch_x = [[self.get_w2v(y) for y in x] for x in batch_x]
        batch_x, batch_m = np.array([[np.zeros(300) for i in range(max_l - len(x[:max_l]))] + x[:max_l] for x in batch_x]), np.array([[[0.] for i in range(max_l - len(x))] + [[1.] for i in range(len(x[:max_l]))] for x in batch_x])
        return batch_x, batch_m, batch_y

    def get_w2v(self, w):
        '''
        Returns a vector for a given word
        
        Parameters
        ----------
        w : str

        Returns
        -------
        w2v: float32 numpy array of dimensions [300]
        '''
        if w in self.word2vec:
            w2v = self.word2vec[w]
        else:
            w2v = np.zeros(300)
        return w2v

def load_data():
    '''
    Loads data

    Returns
    -------
    train: list of training data
    val: list of validation data
    test: list of test data
    word2vec: dict[word] = word2vec(word)
    '''

    data_file = "data/sentiment_dataset.txt"
    w2v_file = "data/sentiment_w2v.txt"
    data1_pos = list()
    data1_neg = list()
    data2_pos = list()
    data2_neg = list()
    data3_pos = list()
    data3_neg = list()
    with open(w2v_file, encoding='latin1') as f:
        word2vec = json.loads(f.read())
    for w in word2vec:
        word2vec[w] = np.array(word2vec[w])
    with open(data_file, encoding='latin1') as f:
        for line in f:
            s, cl = line.replace("\n", "").split("\t")
            x = (s.split(), float(cl))
            if len(data1_pos + data1_neg) < 1000:
                if int(cl):
                    data1_pos.append(x)
                else:
                    data1_neg.append(x)
            elif len(data2_pos + data2_neg) < 1000:
                if int(cl):
                    data2_pos.append(x)
                else:
                    data2_neg.append(x)
            else:
                if int(cl):
                    data3_pos.append(x)
                else:
                    data3_neg.append(x)
    train = data1_pos[:420] + data1_neg[:420] + data2_pos[:420] + data2_neg[:420] + data3_pos[:420] + data3_neg[:420]
    test = data1_pos[420:460] + data1_neg[420:460] + data2_pos[420:460] + data2_neg[420:460] + data3_pos[420:460] + data3_neg[420:460]
    val = data1_pos[460:] + data1_neg[460:] + data2_pos[460:] + data2_neg[460:] + data3_pos[460:] + data3_neg[460:]
    return train, val, test, word2vec
