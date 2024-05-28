from utils.argparser import argparser
from utils.misc import *
from utils.dataset import prepare_dataset
from datetime import datetime
from utils.train import train_model


if __name__ == "__main__":
    # get parameters passed from terminal
    parser = argparser()   
    args = parser.parse_args()   
    logger = set_logger(args.log_path, 'server')
    
    if args.work_type == "gen_data":
        now = datetime.now().strftime("%Y%m%d-%H%M")
        logger.info('start preparing datasets at {}'.format(now))
        prepare_dataset(args, logger)
    else:    
        if args.dataset == 'EMNIST':
            args.num_classes = 62
        elif args.dataset == 'CIFAR100':
            args.num_classes = 100
        elif args.dataset == 'Tiny-ImageNet':
            args.num_classes = 200
        algorithm = args.algorithm
        module_name = os.path.join('arch', str(algorithm))
        module_path = module_name + '.py'
        if not os.path.exists(module_path):
            raise ImportError("{} not being implemented in modules folder yet".format(algorithm))
        else:
            train_model(args, module_name, logger)


        
