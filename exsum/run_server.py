#!python
import argparse, os
import importlib.util
import dill

from exsum import Model
from exsum.gui_server import Server
from argparse import RawTextHelpFormatter

desc = '''
EXSUM_FN specifies the exsum.Model object to be visualized in 
one of two ways: 
  1. It is the path to a python script containing the model 
     object in global namespace, whose name is assumed to be 
     "model" or specified by "MODEL_VAR_NAME" variable. In 
     this case, EXSUM_FN should end with ".py". 
  2. It is the path to a Python pickle file containing the 
     object. Note that since the object contains applicability 
     and behavior function definitions of each rule, it may 
     need to be produced by the "dill" library. In this case, 
     EXSUM_FN should end with ".pkl". 
LOG_DIR specifies the location for the server logs. 
SAVE_DIR specifies the location for the ExSum saves. 
'''

def parse_args():
    parser = argparse.ArgumentParser(description=desc, 
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('exsum_fn', type=str, metavar='EXSUM_FN')
    parser.add_argument('--model-var-name', default='model', type=str)
    parser.add_argument('--log-dir', default='logs', type=str)
    parser.add_argument('--save-dir', default='saves', type=str)
    args = parser.parse_args()
    fn = args.exsum_fn
    assert fn is not None, 'EXSUM_FN not provided'
    assert fn.endswith('.py') or fn.endswith('.pkl'), \
        'EXSUM_FN should be a .py or .pkl file'
    if fn.endswith('.py'):
        spec = importlib.util.spec_from_file_location("module.name", fn)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, args.model_var_name)
    else:
        with open(fn, 'rb') as f:
            model = dill.load(f)
    assert isinstance(model, Model), '"model" variable is not a Model object'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    return model, args.log_dir, args.save_dir

def main():
    model, log_dir, save_dir = parse_args()
    server = Server(model, log_dir=log_dir, save_dir=save_dir)
    server.run(debug=False)

if __name__ == '__main__':
    main()
