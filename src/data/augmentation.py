"""
Data augmentation techniques for multimodal EEG + eye-tracking data.
"""

import numpy as np
import random
from copy import deepcopy
from spektral.data import Graph


class DataAugmenter:
    """
    Data augmentation for multimodal emotion recognition graphs.
    """

    def __init__(self,
                 noise_prob=0.7,
                 temporal_prob=0.5,
                 graph_prob=0.3,
                 noise_std=0.05,
                 time_shift_range=0.1,
                 eye_jitter_std=2.0):
        """
        Initialize augmenter with configurable parameters.

        Args:
            noise_prob (float): Probability of applying noise augmentation
            temporal_prob (float): Probability of applying temporal augmentation
            graph_prob (float): Probability of applying graph augmentation
            noise_std (float): Standard deviation for Gaussian noise (relative to signal)
            time_shift_range (float): Range for temporal shifting (as fraction of length)
            eye_jitter_std (float): Standard deviation for eye-tracking jitter (pixels)
        """
        self.noise_prob = noise_prob
        self.temporal_prob = temporal_prob
        self.graph_prob = graph_prob
        self.noise_std = noise_std
        self.time_shift_range = time_shift_range
        self.eye_jitter_std = eye_jitter_std

    def augment_graph(self, graph):
        """
        Apply augmentation to a single graph.

        Args:
            graph (spektral.data.Graph): Input graph

        Returns:
            spektral.data.Graph: Augmented graph
        """
        # Create copy to avoid modifying original
        aug_graph = deepcopy(graph)

        # Apply augmentations probabilistically
        if random.random() < self.noise_prob:
            aug_graph = self._apply_noise_augmentation(aug_graph)

        if random.random() < self.temporal_prob:
            aug_graph = self._apply_temporal_augmentation(aug_graph)

        if random.random() < self.graph_prob:
            aug_graph = self._apply_graph_augmentation(aug_graph)

        return aug_graph

    def _apply_noise_augmentation(self, graph):
        """Apply noise-based augmentation."""
        features = graph.x.copy()

        # Separate EEG and eye-tracking features
        eye_features = features[:, :7]  # First 7 are eye-tracking
        eeg_features = features[:, 7:]  # Rest are EEG

        # Add Gaussian noise to EEG features (more robust to noise)
        if eeg_features.size > 0:
            eeg_noise = np.random.normal(0, self.noise_std, eeg_features.shape)
            # Scale noise relative to signal magnitude
            eeg_std = np.std(eeg_features, axis=0, keepdims=True)
            eeg_std = np.where(eeg_std == 0, 1, eeg_std)  # Avoid division by zero
            eeg_features += eeg_noise * eeg_std

        # Add smaller jitter to eye-tracking features
        if eye_features.size > 0:
            # Only jitter spatial coordinates (pupil_x, pupil_y)
            if eye_features.shape[1] >= 3:
                xy_noise = np.random.normal(0, self.eye_jitter_std, (eye_features.shape[0], 2))
                # Normalize jitter to reasonable pixel range
                xy_noise = np.clip(xy_noise, -5, 5)
                eye_features[:, 1:3] += xy_noise  # pupil_x, pupil_y

        # Recombine features
        graph.x = np.concatenate([eye_features, eeg_features], axis=1)

        return graph

    def _apply_temporal_augmentation(self, graph):
        """Apply temporal augmentation."""
        features = graph.x.copy()
        n_nodes = features.shape[0]

        if n_nodes <= 1:
            return graph

        # Random temporal operations
        aug_type = random.choice(['shift', 'subsample', 'interpolate'])

        if aug_type == 'shift' and n_nodes > 2:
            # Circular shift nodes (simulates temporal shift)
            shift = random.randint(1, max(1, int(n_nodes * self.time_shift_range)))
            graph.x = np.roll(features, shift, axis=0)

        elif aug_type == 'subsample' and n_nodes > 3:
            # Keep 80-90% of nodes (subsample temporally)
            keep_ratio = random.uniform(0.8, 0.9)
            keep_count = max(2, int(n_nodes * keep_ratio))

            # Sample evenly spaced indices
            indices = np.linspace(0, n_nodes-1, keep_count, dtype=int)
            graph.x = features[indices]

            # Also subsample adjacency matrix
            if hasattr(graph, 'a') and graph.a is not None:
                graph.a = graph.a[np.ix_(indices, indices)]

        elif aug_type == 'interpolate' and n_nodes > 2:
            # Add interpolated nodes between existing ones
            new_features = []
            for i in range(n_nodes - 1):
                new_features.append(features[i])
                # 50% chance to add interpolated point
                if random.random() < 0.3:
                    interpolated = 0.5 * (features[i] + features[i + 1])
                    new_features.append(interpolated)
            new_features.append(features[-1])

            graph.x = np.array(new_features)

        return graph

    def _apply_graph_augmentation(self, graph):
        """Apply graph structure augmentation."""
        if not hasattr(graph, 'a') or graph.a is None:
            return graph

        adj = graph.a.copy()

        # Random graph operations
        aug_type = random.choice(['edge_dropout', 'feature_masking'])

        if aug_type == 'edge_dropout':
            # Randomly drop 10-20% of edges
            dropout_rate = random.uniform(0.1, 0.2)
            mask = np.random.random(adj.shape) > dropout_rate

            # Keep self-loops and ensure connectivity
            np.fill_diagonal(mask, True)
            adj = adj * mask

        elif aug_type == 'feature_masking':
            # Randomly mask some node features
            mask_rate = random.uniform(0.05, 0.15)
            n_nodes, n_features = graph.x.shape

            # Create random mask
            mask = np.random.random((n_nodes, n_features)) > mask_rate
            graph.x = graph.x * mask

        graph.a = adj
        return graph

    def augment_dataset(self, graphs, augmentation_factor=3, preserve_class_balance=True):
        """
        Augment entire dataset.

        Args:
            graphs (list): List of Graph objects
            augmentation_factor (int): How many augmented versions per original
            preserve_class_balance (bool): Whether to balance classes during augmentation

        Returns:
            list: Augmented list of graphs
        """
        augmented_graphs = list(graphs)  # Start with originals

        if preserve_class_balance:
            # Group by class
            class_groups = {}
            for graph in graphs:
                label = graph.y if hasattr(graph, 'y') else 0
                if label not in class_groups:
                    class_groups[label] = []
                class_groups[label].append(graph)

            # Find minority class size
            min_class_size = min(len(group) for group in class_groups.values())
            target_size = min_class_size * (augmentation_factor + 1)

            # Augment each class to target size
            for label, group in class_groups.items():
                current_size = len(group)
                needed = max(0, target_size - current_size)

                for _ in range(needed):
                    # Sample random graph from this class
                    source_graph = random.choice(group)
                    augmented = self.augment_graph(source_graph)
                    augmented_graphs.append(augmented)
        else:
            # Simple augmentation - multiply each graph
            for _ in range(augmentation_factor):
                for graph in graphs:
                    augmented = self.augment_graph(graph)
                    augmented_graphs.append(augmented)

        # Shuffle final dataset
        random.shuffle(augmented_graphs)

        print(f"Dataset augmentation complete:")
        print(f"  Original size: {len(graphs)}")
        print(f"  Augmented size: {len(augmented_graphs)}")
        print(f"  Augmentation factor: {len(augmented_graphs) / len(graphs):.1f}x")

        return augmented_graphs


def create_augmented_dataset(original_graphs, augmentation_config=None):
    """
    Convenience function to create augmented dataset.

    Args:
        original_graphs (list): Original graph list
        augmentation_config (dict): Configuration for augmenter

    Returns:
        list: Augmented graphs
    """
    config = augmentation_config or {}
    augmenter = DataAugmenter(**config)

    return augmenter.augment_dataset(
        original_graphs,
        augmentation_factor=config.get('augmentation_factor', 3),
        preserve_class_balance=config.get('preserve_class_balance', True)
    )