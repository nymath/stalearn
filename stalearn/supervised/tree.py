import typing
import numpy as np
import math

from ..base import ClassifierMixin

from graphviz import Digraph

__all__ = [
    "visualize_tree",
    "entropy_impurity",
    "gini_impurity",
    "DecisionTreeClassifier"
]

def visualize_tree(node, dot=None):
    if dot is None:
        dot = Digraph(comment='Decision Tree')
        
    if node.value is not None:
        # leaf nodes
        label = f'''Class {node.value}
n_samples: {node.n_samples}
impurity: {node.impurity:.3f}
value: {node.value_counts}
'''
        dot.node(f'{id(node)}', label=label, shape='ellipse', color='lightgreen')
    else:
        # root nodes
        label = f'''Feature {node.feature_index} < {node.threshold}
n_samples: {node.n_samples}
impurity: {node.impurity:.3f}
leaves: {node.n_leaves}
value: {node.value_counts}
'''
        dot.node(f'{id(node)}', label=label, shape='q')
        if node.left:
            dot.edge(f'{id(node)}', f'{id(node.left)}', label='True')
            visualize_tree(node.left, dot)
        if node.right:
            dot.edge(f'{id(node)}', f'{id(node.right)}', label='False')
            visualize_tree(node.right, dot)
    
    return dot
            
def entropy_impurity(y, class_weights=None):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))

def gini_impurity(y, class_weights=None):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)

class Pruner:

    @staticmethod
    def _prune(tree, node):
        import copy
        new_tree = copy.deepcopy(tree)
        Pruner._prune_model(new_tree, node)
        return new_tree

    @staticmethod
    def _prune_model(tree, node_to_prune):
        if tree.value is not None: # leaf node
            return None
        Pruner._prune_model(tree.left, node_to_prune)
        Pruner._prune_model(tree.right, node_to_prune)
        tree.n_leaves = tree.left.n_leaves + tree.right.n_leaves
        if tree == node_to_prune:
            tree.left = None
            tree.right = None
            tree.value = np.argmax(tree.value_counts)
            tree.n_leaves = 1
    
    @staticmethod
    def weighted_sum_of_impurities(tree):
        total_sums = tree.n_samples
        result = []

        def _weighted_sum_of_impurities(node):
            if node is None:
                return None

            if node.value is not None:
                result.append(node.impurity * node.n_samples)
            _weighted_sum_of_impurities(node.left)
            _weighted_sum_of_impurities(node.right)
        
        _weighted_sum_of_impurities(tree)

        return np.sum(result) / total_sums

    @staticmethod
    def balance_alpha(after_prune, before_prune):
        if before_prune.n_leaves is None or after_prune.n_leaves is None:
            return np.nan

        if (before_prune.n_leaves - after_prune.n_leaves) > 0:
            return (Pruner.weighted_sum_of_impurities(after_prune) - Pruner.weighted_sum_of_impurities(before_prune)) / (before_prune.n_leaves - after_prune.n_leaves)
        else:
            return np.nan

    @staticmethod
    def _helper(before_prune, nodes):
        if before_prune is None:
            return None
        min_alpha = math.inf
        pruned = None
        for node in nodes:
            after_prune = Pruner._prune(before_prune, node)
            current_alpha = Pruner.balance_alpha(after_prune, before_prune)
            if current_alpha < min_alpha:
                min_alpha = current_alpha
                pruned = after_prune

        if pruned is not None:
            return pruned, min_alpha
        else:
            return before_prune, min_alpha
        
    @staticmethod
    def ccp_pruning_path(tree, nodes):
        before_prune = tree
        ccp_alphas = []
        impurities = []
        while True:
            after_prune, _ = Pruner._helper(before_prune, nodes)
            if _ == math.inf:
                break
            impurities.append(Pruner.weighted_sum_of_impurities(after_prune))
            ccp_alphas.append(_)
            
            before_prune = after_prune
        return {
            "ccp_alphas": np.array(ccp_alphas),
            "impurities": np.array(impurities),
        }
    
    @staticmethod
    def prune_trees(tree, nodes, ccp_alpha):
        before_prune = tree
        while True:
            after_prune, _ = Pruner._helper(before_prune, nodes)
            if _ >= ccp_alpha:
                break
            before_prune = after_prune

        return before_prune
    

class DecisionTreeNode:
    def __init__(self,
        feature_index=None, 
        threshold=None, 
        value=None, 
        left=None, 
        right=None,
        n_samples=None,
        n_leaves=None,
        impurity=None,
        value_counts=None,
    ):
        self.feature_index: int = feature_index
        self.threshold = threshold
        self.value = value
        self.left: DecisionTreeNode = left
        self.right: DecisionTreeNode = right
        self.n_samples: int = n_samples
        self.n_leaves: int = n_leaves
        self.impurity: float = impurity
        self.value_counts = value_counts
    
    def __eq__(self, other):
        conds = []
        conds.append(self.feature_index == other.feature_index)
        conds.append(self.threshold == other.threshold)
        return all(conds)

    def _apply(self, func, result):
        result.append(func(self))
        if self.left is not None:
            self.left._apply(func, result)
        if self.right is not None:
            self.right._apply(func, result)
    
    def apply(self, func):
        result = []
        self._apply(func, result)
        return result


class BaseDecisionTree:
    def __init__(
        self,
        criterion: typing.Literal["gini", "entropy"]="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        # min_weight_fraction_leaf=0.0,
        # max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0
    ):
        """_summary_

        Parameters
        ----------
        criterion : {"gini", "entropy"}, optional
            The function to measure the impurity of data, by default "gini".
        splitter : str, optional
            _description_, by default "best"
        max_depth : _type_, optional
            _description_, by default None
        min_samples_split : int, optional
            _description_, by default 2
        min_samples_leaf : int, optional
            _description_, by default 1
        random_state : _type_, optional
            _description_, by default None
        max_leaf_nodes : _type_, optional
            _description_, by default None
        min_impurity_decrease : float, optional
            _description_, by default 0.0
        class_weight : _type_, optional
            _description_, by default None
        ccp_alpha : float, optional
            _description_, by default 0.0
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        # TODO: implement C4.5
        if self.criterion == "gini":
            self.impurity_func = gini_impurity
        elif self.criterion == "entropy":
            self.impurity_func = entropy_impurity
        self._feature_importance = None
        self.fitted = False
        self._leaves_count = 0

    def _compute_impurity_decrease(self, y, y_left, y_right):
        impurity_before = self.impurity_func(y)
        impurity_left, impurity_right = self.impurity_func(y_left), self.impurity_func(y_right)
        impurity_after = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y)
        impurity_decrease = impurity_before - impurity_after
        return impurity_decrease

    def _split(self, X, y) -> typing.Dict[str: typing.Union[float, int, np.ndarray]]:
        max_impurity_decrease = 0

        if len(X) < self.min_samples_split:
            return {}

        split_info = {}
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                impurity_decrease = self._compute_impurity_decrease(y, y_left, y_right)

                if impurity_decrease > max_impurity_decrease:
                    max_impurity_decrease = impurity_decrease
                    split_info = {
                        "feature_index" : feature_index,
                        "threshold" : threshold,
                        "left_mask" : left_mask,
                        "right_mask" : right_mask,
                        "impurity_decrease": impurity_decrease
                    }

        if max_impurity_decrease > self.min_impurity_decrease:
            return split_info
        else:
            return {}

    def _build_tree(self, X, y, depth) -> DecisionTreeNode:
        # calculate node information
        unique_classes, _value_counts = np.unique(y, return_counts=True)
        value_counts = np.zeros(len(self.classes), dtype=int)
        value_counts[unique_classes] += _value_counts
        n_samples = len(y)
        value = np.bincount(y).argmax() # using 0-1 loss to predict the output
        impurity = self.impurity_func(y) # impurity of this node(if not split)

        # try to split the node
        split_info = self._split(X, y)

        # check whether to split the node
        exceed_max_leaf_nodes = self.max_leaf_nodes is not None and self._leaves_count >= self.max_leaf_nodes
        exceed_max_depth = self.max_depth and depth >= self.max_depth
        do_not_split = not split_info
        already_purified = len(unique_classes) == 1

        if already_purified or exceed_max_depth or exceed_max_leaf_nodes or do_not_split:
            # build a leaf node
            self._leaves_count += 1
            return DecisionTreeNode(
            value=value, 
            n_samples=n_samples, 
            impurity=impurity, 
            n_leaves=1, 
            value_counts=value_counts)
        else:
            # build a root node
            left_mask, right_mask = split_info['left_mask'], split_info['right_mask']
            impurity_decrease = split_info['impurity_decrease']
            self._feature_importance[split_info["feature_index"]] += impurity_decrease * n_samples
            left_node = self._build_tree(X[left_mask], y[left_mask], depth+1)
            right_node = self._build_tree(X[right_mask], y[right_mask], depth+1)
            n_leaves = left_node.n_leaves + right_node.n_leaves
            node = DecisionTreeNode(
                feature_index=split_info['feature_index'], 
                threshold=split_info['threshold'], 
                left=left_node, 
                right=right_node,
                n_samples=n_samples,
                impurity=impurity,
                n_leaves=n_leaves,
                value_counts=value_counts
            )
            self._root_nodes.append(node)
            return node
    
    def fit(self, X, y):
        self._root_nodes = []
        self.classes = np.unique(y)
        self._feature_importance = np.zeros(X.shape[1])
        self._leaves_count = 0
        self.tree_ = self._build_tree(X, y, 0)
        self.prune()
        self.fitted = True
    
    def _predict_sample(self, node, x):
        if not node:
            return None
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)
        
    def predict(self, X):
        return np.array([self._predict_sample(self.tree_, x) for x in X])

    def prune(self):
        assert not self.fitted
        self.tree_ = Pruner.prune_trees(self.tree_, self._root_nodes, self.ccp_alpha)

    def cost_complexity_pruning_path(self, X=None, y=None):
        return Pruner.ccp_pruning_path(self.tree_, self._root_nodes)

    @property
    def feature_importance(self):
        return self._feature_importance / self._feature_importance.sum()
    

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        # min_weight_fraction_leaf=0.0,
        # max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha
        )

    def _predict_proba_sample(self, node, x):
        # If the node is a leaf node, return the normalized value_counts as probabilities
        if node.value is not None:
            # Ensure the probability sums up to 1 by normalizing
            return node.value_counts / node.n_samples
        if x[node.feature_index] < node.threshold:
            return self._predict_proba_sample(node.left, x)
        else:
            return self._predict_proba_sample(node.right, x)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        return np.array([self._predict_proba_sample(self.tree_, x) for x in X])

    def predict_log_proba(self, X):
        """
        Predict log class probabilities for the given input samples X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        log_proba : array of shape (n_samples, n_classes)
            The log class probabilities of the input samples.
        """
        proba = self.predict_proba(X)
        return np.log(proba)