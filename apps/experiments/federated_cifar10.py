import logging
import sys
import random

sys.path.append('../../')

from src.apis.rw import IODict
from src.apis.extensions import Dict
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from libs.model.cv.resnet import resnet56
from torch import nn
import libs.model.cv.cnn
from src.data.data_container import DataContainer
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer
from src.apis import lambdas
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
logger.info('Generating Data --Started')
# client_data = data_loader.cifar10_10shards_100c_600min_600max()
trainers_train = Dict()
test_data = Dict()


def poison(dc: DataContainer, rate):
    total_size = len(dc)
    poison_size = int(total_size * rate)
    for i in range(0, poison_size):
        dc.y[i] = 0 if dc.y[i] != 0 else random.randint(1, 9)


client_data = preload('cifar10', LabelDistributor(100, 5, 550, 600))
poisc = 0
for trainer_id, data in client_data.items():
    data = data.shuffle().as_tensor()
    train, test = data.split(0.7)
    trainers_train[trainer_id] = train
    if poisc < 30:
        poison(test, 0.3)
    test_data[trainer_id] = test
    poisc += 1
logger.info('Generating Data --Ended')


def create_model(name):
    if name == 'resnet':
        return resnet56(10, 3, 32)
    else:
        global client_data
        # cifar10 data reduced to 1 dimension from 32,32,3. cnn32 model requires the image shape to be 3,32,32
        client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
        return libs.model.cv.cnn.Cifar10Model()


initialize_model = create_model('cnn')

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=5, optimizer='sgd',
                               criterion='cel', lr=0.01)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    # client_selector=client_selectors.Random(0.15),
    client_selector=client_selectors.All(),
    trainers_data_dict=trainers_train,
    test_data=test_data,
    initial_model=lambda: initialize_model,
    num_rounds=500,
    accepted_accuracy_margin=0.05,
    desired_accuracy=0.99,
)
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(EMDWeightDivergence(show_plot=0))
federated.add_subscriber(SQLiteLogger(id='mycifar', db_path='perf.db', config='cifar10'))
federated.add_subscriber(Resumable(IODict('./mycache.cs')))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
