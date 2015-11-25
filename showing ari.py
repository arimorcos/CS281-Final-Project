data_butler = character_data_manager(load_path,max_doc_loaded=1e10)

network = theano_lstm()


sequence, target = data_butler.offer_data()
cost = network.adadelta_step(sequence, target)
data_butler.adanvce_sequence()

network.set_parameters(epoch=5)
data_butler.set_index(epoch=5)