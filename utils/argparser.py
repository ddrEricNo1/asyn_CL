import argparse


def argparser():
    """Function to add necessary arguments"""
    parser = argparse.ArgumentParser(description="Asynchronous Parallel Continuous Learning")
    global_config(parser)
    cl_config(parser)
    return parser


def global_config(parser):
    """Function to set configs for global settings"""
    parser.add_argument('--work_type', type=str, default='train', 
                        help='work types, e.g., gen_data, train')
    parser.add_argument('--gpu', type=str, default='0', 
                        help="to set gpu ids to work, split by (,)")
    parser.add_argument('--gpu_mem_multiplier', type=float, default=0.5,
                        help='to set fractions to be used per process')
    parser.add_argument('--seed', type=int, default=777, 
                        help="to set seed for initialization")
    parser.add_argument('--num_class_sets', type=int, default=3, 
                        help='number of class sets')
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--num_tasks', type=int, default=5, 
                        help='number of tasks for the dataset')
    parser.add_argument('--num_time_lines', type=int, default=3, 
                        help='number of time lines for each class set')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help="number of classes for each task")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="to set batch size")
    parser.add_argument('--dataset', type=str, default='EMNIST',
                        help='name of the dataset to be trained')
    parser.add_argument('--num_users', type=int, default=2, 
                        help="number of users: n")
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument('--log_every', type=int, default=1,
                        help='log interval')
    parser.add_argument('--download_every', type=int, default=75,
                       help='download interval')
    parser.add_argument('--upload_every', type=int, default=50, 
                       help='upload interval')

    # for dataset
    parser.add_argument('--raw_data_path', type=str, default='data/raw/', 
                        help='path for the raw data')
    parser.add_argument('--data_path', type=str, default='data/', 
                        help='path for the data')
    parser.add_argument('--output_path', type=str, default='results/',
                        help='path for the output')
    parser.add_argument('--log_path', type=str, default='results/log/', 
                        help='path for the log file')
    return parser


def cl_config(parser):
    """Function to set configs for continuous learning"""
    parser.add_argument('--algorithm', type=str, default='gem', 
                        help='continuous learning algorithm to be used on each client')
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='model to be trained on each client')
    parser.add_argument('--n_memories', type=int, default=256,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', type=float, default=0.5,
                        help='memory strength (meaning depends on memory)')
    return parser