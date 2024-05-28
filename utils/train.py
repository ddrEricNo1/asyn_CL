import importlib
from datetime import datetime


def train_model(args, module_name, logger):
    """Function to train the model.
    Args:
        args: argument options
        module_name: the name of the module to be used for algorithm
        logger: logger file
    """
    now = datetime.now().strftime("%Y%m%d-%H%M")
    logger.info('start asynchronous continual learning training at {}'.format(now))
    module = importlib.import_module(module_name.replace('/', '.'))
    logger.info("choose algorithm {}".format(args.algorithm))
    if args.dataset == 'EMNIST':
        args.input_dim = 28 * 28
    elif args.dataset == 'CIFAR100':
        args.input_dim = 3 * 32 * 32
    elif args.dataset == 'Tiny-ImageNet':
        args.input_dim = 3 * 64 * 64
    args_dict = vars(args)
    logger.info(f"arguments: {args_dict}")
    GEM_class = module.Server(args, logger)
    