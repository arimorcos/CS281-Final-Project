import theano_lstm

if __name__ == "__main__":
    network = theano_lstm.lstm_rnn(300,
                  [(128, 128)],
                  50,
                  log_dir='test_log')
    network.initialize_training_adam()