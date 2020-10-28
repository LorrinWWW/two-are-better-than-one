
from .gru import *



def get_mdrnn_layer(
    config, emb_dim=None, 
    direction='B', norm='', rnn_type='GRU', Md='2d', first_layer=False
):
    '''
    direction = ['', 'B', 'Q']
    norm = ['', 'LN']
    Md = ['2d', '25d']
    first_layer  [True, False]
    '''
    
    _direction = direction
    _norm = norm
    _rnn_type = rnn_type
    if first_layer and Md == '25d':
        _Md = '2d'
    else:
        _Md = Md
        
    cell_class = eval(f"{_norm}{rnn_type}{_Md}Cell")
    layer_class = eval(f"{_direction}{rnn_type}{_Md}Layer")
    
    layer = layer_class(config=config, emb_dim=emb_dim, _Cell=cell_class)
    
    return layer