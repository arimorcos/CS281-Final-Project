import theano_lstm

if __name__ == "__main__":
    network = theano_lstm.lstm_rnn(300,
                  [(128, 128)],
                  50)