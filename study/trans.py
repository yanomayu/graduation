import numpy as np
import chainer
import argparse
from seq2seq import load_vocabulary, Seq2seq

parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
parser.add_argument('SOURCE', help='source sentence list')
parser.add_argument('TARGET', help='target sentence list')
parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
parser.add_argument('TARGET_VOCAB', help='target vocabulary file')

parser.add_argument('--modelsave', '-ms', default='', help='save a snapshot of the training')
parser.add_argument('--unit', '-u', type=int, default=1024,        
                                                help='number of units')            
parser.add_argument('--layer', '-l', type=int, default=3,              
                                                help='number of layers')
parser.add_argument('--source', help='source')

args = parser.parse_args()

# print('outut path = ' + args.modelsave)

source_ids = load_vocabulary(args.SOURCE_VOCAB)                               
target_ids = load_vocabulary(args.TARGET_VOCAB)
    
model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
 
chainer.serializers.load_hdf5(args.modelsave, model)

orig_sources = [model.xp.array([[args.source]])]

# device = -1
# new_sources = [chainer.dataset.to_device(device, x) for x in orig_sources]
# translated = model.translate([model.xp.array(args.source)])[0]

new_sources = orig_sources
translated = model.translate(new_sources)[0]

print(str(translated))
