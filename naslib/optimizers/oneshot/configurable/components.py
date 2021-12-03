from abc import ABCMeta, abstractmethod
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from naslib.optimizers.oneshot.darts.optimizer import MixedOp
from naslib.optimizers.oneshot.drnas.optimizer import DrNASMixedOp
from naslib.optimizers.oneshot.gdas.optimizer import GDASMixedOp


class AbstractGraphModifier(metaclass=ABCMeta):
    @abstractmethod
    def update_graph_edges(self, graph, scope):
        raise NotImplementedError()

    @abstractmethod
    def update_graph_nodes(self, graph, scope):
        raise NotImplementedError()

    def update_graph(self, graph, scope):
        self.update_graph_edges(graph, scope)
        self.update_graph_nodes(graph, scope)

    def new_epoch(self):
        pass


class AbstractEdgeOpModifier(AbstractGraphModifier):
    def update_graph_edges(self, graph, scope):
        pass

    def update_graph_nodes(self, graph, scope):
        pass


class AbstractCombOpModifier(AbstractGraphModifier):
    def update_graph_edges(self, graph, scope):
        pass

    def update_graph_nodes(self, graph, scope):
        pass


class AbstractArchitectureSampler(AbstractGraphModifier):

    @abstractmethod
    def sample_arch_weights(self, graph, scope):
        raise NotImplementedError()

    @abstractmethod
    def remove_sampled_arch_weights(self, graph, scope):
        raise NotImplementedError()

    @abstractmethod
    def get_arch_weights(self, graph):
        raise NotImplementedError

    def _update_ops(self, edge):
        primitives = edge.data.op
        edge.data.set("op", self.__class__.mixed_op(primitives))

    def set_device(self, device):
        self.device = device


class DARTSSampler(AbstractArchitectureSampler):
    mixed_op = MixedOp

    def update_graph_edges(self, graph, scope):
        graph.update_edges(self._add_alphas, scope=scope, private_edge_data=False)
        graph.update_edges(self._update_ops, scope=scope, private_edge_data=True)

    def get_arch_weights(self, graph):
        arch_weights = []
        for alpha in graph.get_all_edge_data('alpha'):
            arch_weights.append(alpha)

        return arch_weights

    def _add_alphas(self, edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            1e-3 * torch.randn(size=[len_primitives], requires_grad=True)
        )
        edge.data.set('alpha', alpha, shared=True)

    def update_graph_nodes(self, graph, scope):
        pass

    def sample_arch_weights(self, graph, scope):
        pass

    def remove_sampled_arch_weights(self, graph, scope):
        pass


class DrNASSampler(DARTSSampler):

    mixed_op = DrNASMixedOp

    def update_graph(self, graph, scope):
        super().update_graph(graph, scope)
        self.anchor = Dirichlet(
            torch.ones_like(
                torch.nn.utils.parameters_to_vector(self.architectural_weights)
            ).to(self.device)
        )

    def sample_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._sample_alphas(edge),
            scope=scope,
            private_edge_data=False,
        )

    def remove_sampled_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=self._remove_sampled_alphas,
            scope=scope,
            private_edge_data=False,
        )

    def _sample_alphas(self, edge):
        beta = F.elu(edge.data.alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        edge.data.set("sampled_arch_weight", weights, shared=True)

    def _remove_sampled_alphas(self, edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")


class GDASSampler(DARTSSampler):

    mixed_op = GDASMixedOp

    def __init__(self, epochs, tau_max, tau_min):
        self.epochs = epochs
        self.tau_max = tau_max
        self.tau_min = tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = torch.Tensor([self.tau_max])

    def new_epoch(self):
        self.tau_curr += self.tau_step

    def update_graph(self, graph, scope):
        super().update_graph(graph, scope)
        graph.register_buffer('tau', self.tau_curr)

    def sample_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=lambda edge: self._sample_alphas(edge),
            scope=scope,
            private_edge_data=False,
        )

    def remove_sampled_arch_weights(self, graph, scope):
        graph.update_edges(
            update_func=self._remove_sampled_alphas,
            scope=scope,
            private_edge_data=False,
        )

    def _sample_alphas(self, edge):
        arch_parameters = torch.unsqueeze(edge.data.alpha, dim=0)

        while True:
            gumbels = -torch.empty_like(arch_parameters).exponential_().log()
            gumbels = gumbels.to(self.device)
            self.tau_curr = self.tau_curr.to(self.device)
            arch_parameters = arch_parameters.to(self.device)
            logits = (arch_parameters.log_softmax(dim=1) + gumbels) / self.tau_curr
            probs = torch.nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        weights = hardwts[0]
        argmaxs = index[0].item()

        edge.data.set("sampled_arch_weight", weights, shared=True)
        edge.data.set("argmax", argmaxs, shared=True)

    def _remove_sampled_alphas(self, edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")


class DummyAbstractArchSampler(AbstractArchitectureSampler):
    def update_graph_nodes(self, graph, scope):
        pass

    def update_graph_edges(self, graph, scope):
        pass

    def get_arch_weights(self, graph):
        pass

    def sample_arch_weights(self, graph, scope):
        pass

    def remove_sampled_arch_weights(self, graph, scope):
        pass

    def new_epoch(self):
        pass