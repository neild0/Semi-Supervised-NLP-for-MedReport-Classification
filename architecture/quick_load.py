from fastai.text import *
from .model import *

def load_ccds_model(text_data:DataBunch=None,
                    labeled_data:DataBunch=None,
                    path_to_lm:str=None,
                    path_to_classifier:str=None,
                    encoding_size:int=400,
                    layer_size:int=1152,
                    num_layers:int=3,
                    pad_token:int=1,
                    hidden_dropout:float=0.2,
                    input_dropout:float=0.6,
                    embed_dropout:float=0.1,
                    weight_dropout:float=0.5,
                    output_dropout:float=0.1,
                    qrnn_cells:bool=False, 
                    bidirectional:bool=False,
                    num_classes:int=2,
                    decoder_layer_sizes:Collection[int]=[50],
                    decoder_dropout:Collection[float]=[0.1],
                    bptt_classifier:int=70,
                    max_length:int=2000):
    
    vocab_size = len(text_data.vocab.itos)
    
    encoder = us_ccds_encoder(vocab_sz=vocab_size,
                          enc_sz=encoding_size, 
                          n_hid=layer_size, 
                          n_layers=num_layers, 
                          pad_token=pad_token, 
                          hidden_p=hidden_dropout,
                          input_p=input_dropout, 
                          embed_p=embed_dropout, 
                          weight_p=weight_dropout, 
                          qrnn=qrnn_cells, 
                          bidir=bidirectional)
    
    
    decoder = us_ccds_decoder(n_out=vocab_size, 
                          n_hid=encoding_size, 
                          output_p=output_dropout, 
                          tie_encoder=encoder.encoder, 
                          bias=True)
    
    custom_model = SequentialRNN(encoder,decoder)
    learn_awd = LanguageLearner(text_data, custom_model)
    learn_awd.model = custom_model.cuda()

    pretrained = learn_awd.model[0]
    encoder = su_ccds_encoder(bptt=bptt_classifier, 
                              max_len=max_length,
                              custom_model=pretrained,
                              pad_idx=pad_token)
    decoder = su_ccds_decoder(num_classes,
                              encoding_sz=encoding_size,
                              lin_ftrs=decoder_layer_sizes,
                              ps=decoder_dropout)
    
    learn = text_classifier(labeled_data,encoder,decoder)
    learn.load(path_to_classifier)
    learn.freeze()
    return learn