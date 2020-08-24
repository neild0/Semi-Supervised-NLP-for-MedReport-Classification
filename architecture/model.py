from fastai.text import *
import pandas as pd

vocab_size = None
encoding_size = None
layer_size = None
num_layers = None

num_classes = None
decoder_layer_sizes = None
decoder_dropout = None
bptt_classifier = None
max_length = None
output_nodes=None

pad_token = None
hidden_dropout = None
input_dropout = None
embed_dropout = None
weight_dropout = None
output_dropout = None
qrnn_cells = None
bidirectional = None

def dropout_mask(x:Tensor, sz:Collection[int], p:float):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(Module):
    def __init__(self, p:float=0.5): self.p=p

    def forward(self, x:Tensor)->Tensor:
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class WeightDropout(Module):
    def __init__(self, module:nn.Module, weight_p:float, layer_names:Collection[str]=['weight_hh_l0']):
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args:ArgStar):
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()

class EmbeddingDropout(Module):
    #Dropout

    def __init__(self, emb:nn.Module, embed_p:float):
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words:LongTensor, scale:Optional[float]=None)->Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def awd_lstm_lm_split(model:nn.Module) -> List[nn.Module]:
    groups = [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    return groups + [[model[0].encoder, model[0].encoder_dp, model[1]]]

def awd_lstm_clas_split(model:nn.Module) -> List[nn.Module]:
    groups = [[model[0].module.encoder, model[0].module.encoder_dp]]
    groups += [[rnn, dp] for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    return groups + [[model[1]]]
    
class us_ccds_encoder(Module):
    #Encoder
    initrange=0.1

    def __init__(self, vocab_sz:int=vocab_size, enc_sz:int=encoding_size, n_hid:int=layer_size, n_layers:int=num_layers, pad_token:int=pad_token, hidden_p:float=hidden_dropout,
                 input_p:float=input_dropout, embed_p:float=embed_dropout, weight_p:float=weight_dropout, qrnn:bool=qrnn_cells, bidir:bool=bidirectional):
        self.bs,self.qrnn,self.enc_sz,self.n_hid,self.n_layers = 1,qrnn,enc_sz,n_hid,n_layers
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, enc_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            from .qrnn import QRNN
            self.rnns = [QRNN(enc_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else enc_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir) 
                         for l in range(n_layers)]
            for rnn in self.rnns: 
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(enc_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else enc_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:Tensor, from_embeddings:bool=False)->Tuple[Tensor,Tensor]:
        if from_embeddings: bs,sl,es = input.size()
        else: bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        nh = (self.n_hid if l != self.n_layers - 1 else self.enc_sz) // self.n_dir
        return one_param(self).new(self.n_dir, self.bs, nh).zero_()

    def select_hidden(self, idxs):
        if self.qrnn: self.hidden = [h[:,idxs,:] for h in self.hidden]
        else: self.hidden = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
        
        
class us_ccds_decoder(Module):
    #Decoder
    initrange=0.1

    def __init__(self, n_out:int=vocab_size, n_hid:int=encoding_size, output_p:float=output_dropout, tie_encoder:nn.Module=None, bias:bool=True):
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output)
        return decoded, raw_outputs, outputs
        
def masked_concat_pool(outputs, mask):
    output = outputs[-1]
    avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
    avg_pool *= output.size(1) / (output.size(1)-mask.type(avg_pool.dtype).sum(dim=1))[:,None]
    max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
    x = torch.cat([output[:,-1], max_pool, avg_pool], 1)
    return x
    
    
class MultiBatchEncoder(Module):
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=1):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx

    def concat(self, arrs:Collection[Tensor])->Tensor:
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o = self.module(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_len):
                masks.append(input[:,i: min(i+self.bptt, sl)] == self.pad_idx)
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1)
        
        
def su_ccds_encoder(bptt:int=None, 
                    max_len:int=None,
                    custom_model=None,
                    pad_idx=1) -> nn.Module:
    
    encoder = MultiBatchEncoder(bptt, max_len, custom_model.cuda(), pad_idx=pad_idx)
    return encoder
    
    
def su_ccds_decoder(n_class:int=num_classes, 
                    lin_ftrs:Collection[int]=decoder_layer_sizes,
                    encoding_sz:int=encoding_size,
                    ps:Collection[float]=decoder_dropout) -> nn.Module:
    if ps is None:  
        ps = [0.1]*len(lin_ftrs)
    layers = [encoding_sz * 3] + lin_ftrs + [n_class]
    ps = [0.1] + ps
    decoder = PoolingLinearClassifier(layers, ps)
    return decoder
    
def text_classifier(data:DataBunch,
                            encoder=None,
                            decoder=None,
                            **learn_kwargs) -> 'TextClassifierLearner':
    model = SequentialRNN(encoder, decoder)
    learn = RNNLearner(data, model, split_func=awd_lstm_clas_split, **learn_kwargs)
    return learn
    

