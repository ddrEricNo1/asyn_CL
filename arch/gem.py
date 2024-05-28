from datetime import datetime
import torch
import numpy as np
from utils.misc import *
import random
import multiprocessing as mp
import threading
import time
from torchvision.datasets import ImageFolder
from utils.dataset import emnist_transform, imagenet_transform
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import quadprog
from ctypes import c_int
from signal import signal, SIGPIPE, SIG_DFL


signal(SIGPIPE, SIG_DFL)


class Server:
    def __init__(self, args, logger):
        self.args = args
        self.gpu = [int(i) for i in args.gpu.split(',')]
        self.logger = logger
        self.logger.info("is initializing the server")
        self.workerObj = Worker

        # GEM settings
        self.n_memories = args.n_memories   # number of memories per task
        self.input_dim = args.input_dim
        self.num_tasks = args.num_tasks

        self.grad_dim = []  # number of all the parameters each layer

        # allocate counters
        manager = mp.Manager()
        self.observed_tasks = manager.list()    # observed tasks so far

        self.worker_to_task = mp.Array(c_int, self.args.num_users)    # 实时更新每一个worker正在工作的任务名称
        for i in range(self.args.num_users):
            self.worker_to_task[i] = -1

        self.dataset_name = args.dataset

        if args.model == 'MLP':
            from modules.MLP import MLP
            self.model = MLP(self.input_dim, args.num_classes)
        elif args.model == 'resnet18':
            from modules.resnet import ResNet18
            self.model = ResNet18(self.args.num_classes)

        self.model.initialize()
        self.model.share_memory()

        self.init_memories()

        # 定义活跃的worker数目, 最终一个位置判断是否所有任务都已经分发完毕
        active_workers = mp.Array('b', [False] * (self.args.num_users + 1))
        task_queue = mp.Queue() # task queue for all the processes

        processes = self.init_workers(active_workers, task_queue)

        # 创建一个子线程用于随机加入任务
        task_generate = threading.Thread(target=self.add_tasks, args=(task_queue, active_workers))
        task_generate.start()

        for p in processes:
            p.join()
        
        self.logger.info("Server has finished the training on dataset {}".format(self.args.dataset))
    
    def init_workers(self, active_workers, task_queue):
        processes = []
        for idx, gpu_id in enumerate(self.gpu):
            workerObj = Worker(self.args, gpu_id, self.model, active_workers, self.memory_data, self.memory_labels, 
                            self.grad_dim, task_queue, self.observed_tasks, self.worker_to_task)
            self.logger.info("creating worker {}".format(gpu_id))
            p = mp.Process(target=self.invoke_worker, args=(workerObj, ))
            processes.append(p)
            self.logger.info("start worker {}".format(idx))
            p.start()
        return processes

    def invoke_worker(self, workerObj):
        """Function to invoke the worker"""
        workerObj.train()

    def add_tasks(self, task_queue, active_workers):
        """Function to add tasks sequentially. This function makes sure that every second there will be at least one task sent on the workers."""
        remaining = set([i for i in range(self.num_tasks)]) # 剩余没有到来的task编号
        while len(remaining) != 0:
            add_flag = random.random() < 0.5
            # 如果所有worker都空闲状态, 则必须添加一个新的任务进入
            if not any(active_workers):
                add_flag = True

            if add_flag:
                task_id = remaining.pop()
                task_queue.put(task_id)
                now = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.logger.info("task {} is added into the task queue at {}".format(task_id, now))

            time.sleep(60)
        
        active_workers[self.args.num_users] = True
        self.logger.info("Successfully add all the tasks into the queue")

    def init_memories(self):
        """Function to init memories for past task data and label"""
        for param in self.model.parameters():
            self.grad_dim.append(param.data.numel())

        # episodic memory 
        self.memory_data = torch.zeros(size=(self.num_tasks, self.n_memories, self.input_dim), dtype=torch.float32).share_memory_()

        # label for episodic label
        self.memory_labels = torch.zeros((self.num_tasks, self.n_memories), dtype=torch.int).share_memory_()

        self.logger.info(f'Successfully initialize memories, data: {self.memory_data.size()}, label: {self.memory_labels.size()}, number of parameters per task: {sum(self.grad_dim)}')


class Worker:
    def __init__(self, args, gpu_id, global_model, active_workers, episodic_memories, episodic_labels, grad_dim, task_queue, observed_tasks, worker_to_task):
        """Worker class for asynchronous continual learning.
        Args:
            args: argument list
            gpu_id: gpu id
            global_model: weights of global mdoel
            active_workers: list of flags indicating whether workers are working or not
            episodic_memories: episodic memories for observed tasks
            episodic_labels: labels for episodic memories
            grads: gradients of the past task
            grad_dim: number of parameters per layer for each model
            task_queue: queue for tasks shared by all the processes
            observed_tasks: observed task sequences
        """
        self.args= args
        self.gpu_id = gpu_id
        self.active_workers = active_workers
        self.global_model = global_model
        self.episodic_memories = episodic_memories  # 存储属于过去task的训练样本
        self.episodic_labels = episodic_labels  # 存储过去task样本对应的label
        self.grad_dim = grad_dim
        self.task_queue = task_queue
        self.observed_tasks = observed_tasks
        self.worker_to_task = worker_to_task

        self.old_task = -1  # 该进程中上一个遇到的task identifier
        self.mem_cnt = 0
        self.device = torch.device('cuda:{}'.format(gpu_id))
        self.margin = args.memory_strength
        self.n_memories = args.n_memories

        self.grads = torch.zeros(size=(sum(self.grad_dim), self.args.num_tasks), dtype=torch.float32)   # 存储每个任务的梯度

        if args.model == 'MLP':
            from modules.MLP import MLP
            # self.model = MLP(self.args.input_dim, self.args.num_classes).to(self.device)
            self.model = MLP(self.args.input_dim, self.args.num_classes)
        elif args.model == 'resnet18':
            from modules.resnet import ResNet18
            # self.model = ResNet18(self.args.num_classes).to(self.device)
            self.model = ResNet18(self.args.num_classes)

        self.logger = set_logger(args.log_path, 'worker_{}'.format(self.gpu_id))

        self.load_weight()

        # for training
        self.ce = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999))

    def load_weight(self):
        """Function to load global weights into the local model."""
        self.logger.info(f"worker {self.gpu_id} has successfully downloaded weight from the global model")
        self.model.load_state_dict(self.global_model.state_dict())
    
    def forward(self, x, target_idx):
        out = self.model(x)
        for i in range(out.shape[1]):
            if i not in target_idx:
                out[:, i].data.fill_(-10e10)
        return out

    def upload_weight(self):
        """Function to upload the local weight into server model."""
        self.logger.info(f"worker {self.gpu_id} has successfully uploaded weight to the global model")
        self.global_model.load_state_dict(self.model.state_dict())
        
    def load_task(self, t):
        """Function to load train and test dataset for task t.
        Args:
            t: task identifier
        """
        path = os.path.join(self.args.data_path, "{}_task".format(self.args.dataset))
        path = os.path.join(path, str(self.args.count))
        real_path = os.path.join(path, "task_{}".format(t + 1))
        if self.args.dataset == 'EMNIST':
            dataset_train = ImageFolder(root=real_path + '/train', transform=emnist_transform)
            dataset_test = ImageFolder(root=real_path + '/test', transform=emnist_transform)
        elif self.args.dataset == 'CIFAR100':
            train_data, train_label = torch.load(real_path + 'train.pt')
            test_data, test_label = torch.load(real_path + 'test.pt')
            dataset_train = TensorDataset(train_data, train_label)
            dataset_test = TensorDataset(test_data, test_label)
        elif self.args.dataset == 'Tiny-ImageNet':
            dataset_train = ImageFolder(root=real_path + '/train', transform=imagenet_transform)
            dataset_test = ImageFolder(root=real_path + '/test', transform=imagenet_transform)

        return dataset_train, dataset_test

    def train(self):
        """Function to train the worker."""
        # 当还有任务没有加入到队列中，或者任务队列不为空时
        while not self.active_workers[self.args.num_users] or not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                self.worker_to_task[self.gpu_id] = task
                self.active_workers[self.gpu_id] = True
            except Exception:
                time.sleep(5)
                continue
            # 读取新的server中模型数据
            self.load_weight()

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logger.info('{} is extracted from queue by worker {} at {}'.format(task, self.gpu_id, now))

            dataset_train, dataset_test = self.load_task(task)
            train_loader = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size)
            test_loader = DataLoader(dataset=dataset_test, batch_size=self.args.batch_size)

            # 将此task的label转化成原始数据集中对应的label的index
            target_idx, _ = compute_target_idx(self.args.raw_data_path, self.args.dataset, dataset_train)

            self.model.train()

            count = 0
            for data, label in train_loader:
                if count % self.args.log_every == 0 and count != 0:
                    self.eval_tasks(task)
                loss = self.observe(data, task, label, target_idx, dataset_train)
                if count % self.args.log_every == 0 and count != 0:
                    self.logger.info(f"worker {self.gpu_id} is working on task {task}, the loss is {loss}")
                if count % self.args.download_every == 0 and count != 0:
                        self.load_weight()
                if count % self.args.upload_every == 0 and count != 0:
                    self.upload_weight()
                    
                count += 1
            
            self.logger.info(f"worker {self.gpu_id} has finished task {task}")
            self.eval_task(test_loader, target_idx, task)

            self.upload_weight()
            self.active_workers[self.gpu_id] = False
            self.worker_to_task[self.gpu_id] = -1
    
    def eval_tasks(self, task):
        """Function to evaluate the model's performance on all the previous observed tasks"""
        self.model.eval()
        idx = self.observed_tasks.index(task)

        # 还需要过滤掉其他worker正在执行的任务

        tasks = set(self.observed_tasks[: idx])
        working_tasks_set = set()
        for i in range(self.args.num_users):
            working_tasks_set.add(self.worker_to_task[i])

        tasks = tasks.difference(working_tasks_set)

        self.logger.info(f"worker {self.gpu_id} is training on task {task}, past tasks are {tasks}")
        results = []

        # 在加入此任务之前，所有其他已经见过的任务
        for task in tasks:
            _, test_dataset = self.load_task(task)
            loader = DataLoader(test_dataset, batch_size=self.args.batch_size)
            target_idx, original_class = compute_target_idx(self.args.raw_data_path, self.args.dataset, test_dataset)

            rt = 0
            for data, label in loader:
                # data = data.to(self.device)
                data = data
                # label = label.to(self.device)
                label = label
                _, pb = torch.max(self.forward(data, target_idx).data.cpu(), dim=1, keepdim=False)
                rt += (pb == convert_label_back(test_dataset, original_class, label)).float().sum()
            
            results.append(rt / len(test_dataset))
        self.logger.info(f'worker {self.gpu_id} is working on task {task}. So far, the model has met tasks: {tasks}, Accuracies on previous tasks are: {results}')

    def eval_task(self, test_loader, target_idx, task):
        """Function to evaluate the model on one test dataset immediately after the training on that task.
        Args:
            test_loader: dataloader of the current task
            target_idx: needed indices of the output
        """
        self.model.eval()
        rt = 0
        count = 0
        for data, label in test_loader:
            # data = data.to(self.device)
            data = data
            # label = label.to(self.device)
            label = label
            _, pb = torch.max(self.forward(data, target_idx).data.cpu(), dim=1, keepdim=False)
            rt += (pb == label).float().sum()
            count += 1

        rt = rt / (count * self.args.batch_size)
        self.logger.info(f'worker {self.gpu_id} accuracy on current task {task} is {rt}')

    def observe(self, x, t, y, target_idx, dataset):
        """Function to train GEM on the local model and update the global memory.
        Args:
            x: data
            t: task identifier
            y: label
            target_idx: needed indices for output
        """
        self.model.train()

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # update ring buffer storing examples from current task
        bsz = y.data.size(0)    # batch size
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.episodic_memories[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])

        if bsz == 1:
            self.episodic_labels[t, self.mem_cnt] = y.data[0] 
        else:
            self.episodic_labels[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradients on previous tasks
        idx = self.observed_tasks.index(t)
        tasks = set(self.observed_tasks[: idx])
        working_tasks_set = set()
        for i in range(self.args.num_users):
            working_tasks_set.add(self.worker_to_task[i])

        tasks = tasks.difference(working_tasks_set)

        self.logger.info(f"worker {self.gpu_id} is training on task {t}, past tasks are {tasks}")
        for past_task in tasks:
            self.model.zero_grad()

            # fwd/bwd on the examples in the memory
            if self.args.dataset == 'EMNIST':
                past_dataset = ImageFolder(os.path.join(self.args.data_path, '{}_task'.format(self.args.dataset)) + '/{}/task_{}/test/'.format(self.args.count, past_task), transform=emnist_transform)
            elif self.args.dataset == 'Tiny-ImageNet':
                past_dataset = ImageFolder(os.path.join(self.args.data_path, '{}_task'.format(self.args.dataset)) + '/{}/{}/test/'.format(self.args.count, past_task), transform=imagenet_transform)
            elif self.args.dataset == 'CIFAR100':
                data, label = torch.load(os.path.join(self.args.data_path, '{}_task'.format(self.args.dataset)) + '/{}/{}/'.format(self.args.count, past_task))
                past_dataset = TensorDataset(data, label)
            past_idx, _ = compute_target_idx(self.args.raw_data_path, self.args.dataset, past_dataset)

            # data = self.episodic_memories[past_task].to(self.device)
            data = self.episodic_memories[past_task]
            # label = self.episodic_labels[past_task].to(self.device)
            label = self.episodic_labels[past_task]

            label = torch.tensor(label, dtype=torch.long)
            past_idx = torch.tensor(past_idx)
            ptloss = self.ce(
                self.forward(data, past_idx).index_select(1, past_idx),
                label
            )

            ptloss.backward()

            self.store_grad(past_task)

        self.model.zero_grad()

        target_idx, _ = compute_target_idx(self.args.raw_data_path, self.args.dataset, dataset)
        target_idx = torch.tensor(target_idx)
        # x = x.to(self.device)
        x = x
        # y = y.to(self.device)
        y = y

        y = torch.tensor(y, dtype=torch.long)
        loss = self.ce(self.forward(x, target_idx).index_select(1, target_idx), y)

        loss.backward()

        if len(tasks) > 0:
            # copy graident of the current task
            self.store_grad(t)

            index = torch.tensor(np.array(list(tasks)), dtype=torch.int32)
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), 
                            self.grads.index_select(1, index))
            
            if (dotp < 0).sum() != 0:
                self.project2cone2(self.grads[:, t].unsqueeze(1),
                                   self.grads.index_select(1, index))

                # copy gradient back
                self.overwrite_grad(self.model.parameters, self.grads[:, t])
        
        self.optim.step()
        return loss

    def overwrite_grad(self, pp, newgrad):
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dim[: cnt])
                en = sum(self.grad_dim[: cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1
    
    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """Function to solve the GEM dual QP described in paper."""
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())  

        P = 0.5 * (P + P.transpose()) + np.eye(P.shape[0]) * 0.005
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin

        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np

        gradient.copy_(torch.Tensor(x).view(-1, 1))

    def store_grad(self, tid):
        """Function to store parameter gradients of past tasks.
        Args:
            tid:task id
        """
        self.grads[:, tid].fill_(0.0)
        cnt = 0
        for param in self.model.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dim[: cnt])
                en = sum(self.grad_dim[: cnt + 1])
                self.grads[beg: en, tid].copy_(param.grad.data.view(-1))
            cnt += 1