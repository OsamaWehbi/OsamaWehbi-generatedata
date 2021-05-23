import copy
from functools import reduce
from src import tools
from src.data.data_container import DataContainer
from src.federated.events import Events, FederatedEventPlug
from src.federated.protocols import Trainer, Aggregator, ClientSelector, ModelInfer


class FederatedLearning:

    def __init__(self, trainer: Trainer, aggregator: Aggregator, client_selector: ClientSelector,
                 tester: ModelInfer, trainers_data_dict: {int: DataContainer}, create_model: callable, num_rounds=10,
                 desired_accuracy=0.9, train_ratio=0.8, ignore_acc_decrease=False, **kwargs):
        self.trainer = trainer
        self.aggregator = aggregator
        self.client_selector = client_selector
        self.tester = tester
        self.ignore_acc_decrease = ignore_acc_decrease
        self.trainers_data_dict = trainers_data_dict
        self.desired_accuracy = desired_accuracy
        self.create_model = create_model
        self.train_ratio = train_ratio
        self.num_rounds = num_rounds
        self.args = kwargs
        self.events = {}
        self.check_params()
        self.context = FederatedLearning.Context(self)

    def start(self):
        self.broadcast(Events.ET_FED_START, **self.configs())
        self.context.reset()
        self.broadcast(Events.ET_INIT, global_model=self.context.model)
        while True:
            self.broadcast(Events.ET_ROUND_START, round=self.context.round_id)
            trainers_ids = self.client_selector.select(list(self.trainers_data_dict.keys()), self.context.round_id)
            self.broadcast(Events.ET_TRAINER_SELECTED, trainers_ids=trainers_ids)
            selected_trainers = tools.dict_select(trainers_ids, self.trainers_data_dict)
            trainers_train_data, trainers_test_data = self.split(selected_trainers)
            self.broadcast(Events.ET_TRAIN_START, trainers_data=trainers_train_data)
            trainers_weights, sample_size_dict = self.train(trainers_train_data)
            self.broadcast(Events.ET_TRAIN_END, trainers_weights=trainers_weights, sample_size=sample_size_dict)
            global_weights = self.aggregator.aggregate(trainers_weights, sample_size_dict, self.context.round_id)
            tools.load(self.context.model, global_weights)
            self.broadcast(Events.ET_AGGREGATION_END, global_weights=global_weights, global_model=self.context.model)
            accuracy, loss, local_acc, local_loss = self.infer(self.context.model, trainers_test_data)
            self.broadcast(Events.ET_ROUND_FINISHED, round=self.context.round_id, accuracy=accuracy, loss=loss,
                           local_acc=local_acc, local_loss=local_loss)
            self.context.new_round()
            if self.context.stop(accuracy):
                self.broadcast(Events.ET_FED_END, aggregated_model=self.context.model)
                break
        return self.context.model

    def train(self, trainers_train_data):
        trained_clients_model = {}
        clients_sample_size = {}
        for trainer_id, train_data in trainers_train_data.items():
            model_copy = copy.deepcopy(self.context.model)
            trained_model = self.trainer.train(model_copy, train_data, self.context)
            sample_size = len(train_data)
            trained_clients_model[trainer_id] = trained_model
            clients_sample_size[trainer_id] = sample_size
        return trained_clients_model, clients_sample_size

    def infer(self, model, trainers_data):
        local_accuracy = {}
        local_loss = {}
        for trainer_id, test_data in trainers_data.items():
            acc, loss = self.tester.infer(model, test_data)
            local_accuracy[trainer_id] = acc
            local_loss[trainer_id] = loss
        total_accuracy = sum(local_accuracy.values()) / len(local_accuracy)
        total_loss = sum(local_loss.values()) / len(local_loss)
        return total_accuracy, total_loss, local_accuracy, local_loss

    def split(self, trainers_data: {int: DataContainer}):
        train_trainers_data = {}
        test_trainers_data = {}
        for trainer_id, data in trainers_data.items():
            train_data, test_data = data.split(self.train_ratio)
            train_trainers_data[trainer_id] = train_data
            test_trainers_data[trainer_id] = test_data
        return train_trainers_data, test_trainers_data

    def check_params(self):
        pass

    def configs(self):
        named = {
            "trainer": self.trainer,
            "aggregator": self.aggregator,
            "client_selector": self.client_selector,
            "ignore_acc_decrease": self.ignore_acc_decrease,
            "trainers_data_dict": self.trainers_data_dict,
            "desired_accuracy": self.desired_accuracy,
            "create_model": self.create_model,
            "train_ratio": self.train_ratio,
            "num_rounds": self.num_rounds
        }
        return reduce(lambda x, y: dict(x, **y), (named, self.args))

    def broadcast(self, event_name, **kwargs):
        if event_name in self.events:
            for item in self.events[event_name]:
                item(kwargs)

    def register_event(self, event_name, action):
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(action)

    def plug(self, plugin: FederatedEventPlug):
        events = plugin.as_events()
        for event_name, call in events.items():
            if plugin.only is not None and event_name not in plugin.only:
                if event_name not in plugin.force():
                    continue
            self.register_event(event_name, call)

    class Context:
        def __init__(self, federated):
            self.round_id = 0
            self.model = None
            self.federated: FederatedLearning = federated

        def new_round(self):
            self.round_id += 1

        def stop(self, acc):
            fl = self.federated
            return (0 < fl.num_rounds <= self.round_id) or acc >= fl.desired_accuracy

        def reset(self):
            self.round_id = 0
            self.model = self.federated.create_model()
