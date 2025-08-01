# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Sequence

from torch.utils.data import IterableDataset

from litgpt.litdata.streaming.dataset import StreamingDataset
from litgpt.litdata.utilities.env import _WorkerEnv

__NUM_SAMPLES_YIELDED_KEY__ = "__NUM_SAMPLES_YIELDED__"
__SAMPLES_KEY__ = "__SAMPLES__"


class CombinedStreamingDataset(IterableDataset):
    """The `CombinedStreamingDataset` enables to stream data from multiple StreamingDataset with the sampling ratio of
    your choice.

    Addtionally, the `CombinedStreamingDataset` keeps track of the number of samples fetched to enable resumability
    of the datasets.

    Note that due to the random sampling, the number of samples returned from the iterator is variable and a function
    of the given seed. The combined dataset will raise a StopIteration as soon as any of the datasets is exhausted.

    """

    def __init__(
        self,
        datasets: List[StreamingDataset],
        seed: int = 42,
        weights: Optional[Sequence[float]] = None,
        iterate_over_all: bool = True,
        sequential_sampling: bool = False,  #todo: check whether this is correct
    ) -> None:
        """
        Arguments:
            datasets: The list of the StreamingDataset to use.
            seed: The random seed to initialize the sampler
            weights: The sampling ratio for the datasets
            iterate_over_all: When iterate_over_all is True, the combined dataset iterates over all the datasets.
                Otherwise, it stops as soon as one raises a StopIteration.
            sequential_sampling: If True, exhaust the first dataset completely before moving to the next, etc.
        """

        self._check_datasets(datasets)

        if iterate_over_all and weights:
            raise ValueError(
                "When `iterate_over_all` is set to True, the weights argument shouldn't be provided.",
                " Instead, it will be computed from the inverse of the dataset length.",
            )
        
        if sequential_sampling and not iterate_over_all:
            raise ValueError(
                "When `sequential_sampling` is set to True, the `iterate_over_all` argument should be set to True.",
            )
            
        self._seed = seed
        self._datasets = datasets
        self._weights = None

        self._iterate_over_all = iterate_over_all
        self._sequential_sampling = sequential_sampling

        if weights is None:
            # Weighted based on the dataset length
            dataset_lens = [len(d) for d in datasets]
            total_len = sum(dataset_lens)
            assert total_len > 0
            self._weights = [len / total_len for len in dataset_lens]
        else:
            weights_sum = sum(weights)
            self._weights = [w / weights_sum for w in weights]

        self._iterator: Optional[_CombinedDatasetIterator] = None
        self._use_streaming_dataloader = False
        self._num_samples_yielded = None
        self._current_epoch = 0
        self.num_workers = 1
        self.batch_size = 1

    def get_len(self, num_workers: int, batch_size: int) -> Optional[int]:
        self.num_workers = num_workers
        self.batch_size = batch_size
        if self._iterate_over_all:
            return self._get_total_length()
        return None

    def __len__(self) -> Optional[int]:
        return self.get_len(1, 1)

    # total length of the datasets
    def _get_total_length(self) -> int:
        return sum(self._get_len(d) for d in self._datasets)

    def _get_len(self, d: Any) -> int:
        if isinstance(d, StreamingDataset):
            return d.get_len(self.num_workers, self.batch_size)
        return len(d)

    def set_epoch(self, current_epoch: int) -> None:
        """Set the current epoch to the datasets on epoch starts.

        When using the StreamingDataLoader, this is done automatically

        """
        self._current_epoch = current_epoch
        for dataset in self._datasets:
            dataset.set_epoch(current_epoch)

    def set_shuffle(self, shuffle: bool) -> None:
        """Set the current shuffle to the datasets."""
        for dataset in self._datasets:
            dataset.set_shuffle(shuffle)

    def set_batch_size(self, batch_size: int) -> None:
        """Set the current batch size to the datasets."""
        self.batch_size = batch_size
        for dataset in self._datasets:
            dataset.set_batch_size(batch_size)

    def set_num_workers(self, num_workers: int) -> None:
        """Set the current number of workers to the datasets."""
        for dataset in self._datasets:
            dataset.set_num_workers(num_workers)

    def set_drop_last(self, drop_last: bool) -> None:
        """Set the current drop_last to the datasets."""
        for dataset in self._datasets:
            dataset.set_drop_last(drop_last)

    def _check_datasets(self, datasets: List[StreamingDataset]) -> None:
        if any(not isinstance(d, StreamingDataset) for d in datasets):
            raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")

    def _set_use_streaming_dataloader(self, use_streaming_dataloader: bool) -> None:
        # Used to prevent returning num_samples_yielded when using PyTorch DataLoader
        self._use_streaming_dataloader = use_streaming_dataloader

    def __iter__(self) -> Iterator[Any]:
        assert self._weights

        worker_env = _WorkerEnv.detect()

        num_samples_yielded = None

        if self._num_samples_yielded is not None and worker_env.rank in self._num_samples_yielded:
            assert isinstance(self._num_samples_yielded, dict)
            num_samples_yielded = self._num_samples_yielded.get(worker_env.rank, 0)

        self._iterator = _CombinedDatasetIterator(
            self._datasets,
            self._seed,
            self._weights,
            self._use_streaming_dataloader,
            num_samples_yielded,
            self._iterate_over_all,
        )
        return self._iterator

    def state_dict(
        self, num_workers: int, batch_size: int, num_samples_yielded: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        if self._iterator is None:
            if num_samples_yielded is None:
                return {}
            return _state_dict(self._datasets, num_samples_yielded, num_workers, batch_size)
        return self._iterator.state_dict(num_workers, batch_size)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return

        if len(state_dict["dataset"]) != len(self._datasets):
            raise RuntimeError(f"The provided state doesn't match the current number of datasets: {self._datasets}.")

        for dataset_idx, dataset in enumerate(self._datasets):
            if str(dataset_idx) not in state_dict["dataset"]:
                raise RuntimeError(f"The provided state doesn't contain the index {dataset_idx}.")

            dataset.load_state_dict(state_dict["dataset"][str(dataset_idx)])

        # Used to iterate over the sampler to avoid sampling the same samples
        if self._use_streaming_dataloader:
            self._num_samples_yielded = state_dict["num_samples_yielded"]


class _CombinedDatasetIterator(Iterator):
    def __init__(
        self,
        datasets: List[StreamingDataset],
        seed: int,
        weights: Sequence[Optional[float]],
        use_streaming_dataloader: bool,
        num_samples_yielded: Any,
        iterate_over_all: bool = False,
        sequential_sampling: bool = False,
    ) -> None:
        self._datasets = datasets
        self._dataset_iters = [iter(dataset) for dataset in datasets]
        self._dataset_indexes: List[Optional[int]] = list(range(len(datasets)))
        self._num_samples_yielded = num_samples_yielded or [0 for _ in range(len(datasets))]
        self._original_weights = deepcopy(weights)
        self._weights = deepcopy(weights)
        self._rng = random.Random(seed)
        self._iterate_over_all = iterate_over_all
        self._is_done = False
        
        self._sequential_sampling = sequential_sampling
        self._current_sequential_index = 0

        if num_samples_yielded is not None:
            self._num_samples_yielded = num_samples_yielded
            for _ in range(sum(num_samples_yielded)):
                choice_indexes: List[int] = [index for index in self._dataset_indexes if index is not None]
                choice_weights: List[float] = [w for w in self._weights if w is not None]
                self._rng.choices(choice_indexes, weights=choice_weights, k=1)

        self._use_streaming_dataloader = use_streaming_dataloader
        self._is_done = False

    def __next__(self) -> Any:
        if self._iterate_over_all and self._sequential_sampling:
            # Sequential approach: exhaust the first dataset completely before moving to the next
            while self._current_sequential_index < len(self._datasets):
                # If there's a dataset index that is None, it means it's been exhausted
                if self._dataset_indexes[self._current_sequential_index] is None:
                    # Move to the next dataset
                    self._current_sequential_index += 1
                    continue

                try:
                    # Fetch sample from the current dataset index
                    return self._get_sample(self._current_sequential_index)
                except StopIteration:
                    # Mark current dataset as exhausted
                    self._dataset_indexes[self._current_sequential_index] = None
                    # Move to the next dataset
                    self._current_sequential_index += 1

            # If we are here, it means all datasets are exhausted
            raise StopIteration
        
        elif self._iterate_over_all:
            while True:
                try:
                    indexes_left = [index for index in self._dataset_indexes if index is not None]
                    if len(indexes_left) > 1:
                        dataset_index = self._get_dataset_index()
                    elif len(indexes_left) == 1:
                        dataset_index = indexes_left[0]
                    return self._get_sample(dataset_index)
                except StopIteration as e:
                    if len(indexes_left) == 1:
                        # If there's only one dataset left and this dataset raises a StopIteration, we are done
                        self._dataset_indexes = list(range(len(self._datasets)))
                        self._weights = deepcopy(self._original_weights)
                        raise e

                    self._dataset_indexes[dataset_index] = None
                    self._weights[dataset_index] = None  # type: ignore
                    new_sum = sum([w for w in self._weights if w is not None])
                    self._weights = [None if w is None else w / new_sum for w in self._weights]

        # stop on the first iteration
        return self._get_sample(self._get_dataset_index())

    def _get_dataset_index(self) -> int:
        # randomly select a dataset index
        indexes = [index for index in self._dataset_indexes if index is not None]
        weights = [w for w in self._weights if w is not None]
        (dataset_index,) = self._rng.choices(indexes, weights=weights, k=1)
        return dataset_index

    def _get_sample(self, dataset_index: int) -> Any:
        # get the sample
        sample = next(self._dataset_iters[dataset_index])

        # keep track the sample was fetched
        self._num_samples_yielded[dataset_index] += 1

        # return a new sample
        if self._use_streaming_dataloader:
            return {
                __SAMPLES_KEY__: sample,
                __NUM_SAMPLES_YIELDED_KEY__: self._num_samples_yielded,
            }
        return sample

    def state_dict(self, num_workers: int = 0, batch_size: int = 1) -> Dict[str, Any]:
        return _state_dict(self._datasets, self._num_samples_yielded, num_workers, batch_size)


def _state_dict(
    datasets: List[StreamingDataset], num_samples_yielded: List[int], num_workers: int = 0, batch_size: int = 1
) -> Dict[str, Any]:
    return {
        str(dataset_idx): dataset.state_dict(
            num_samples_yielded=num_samples_yielded[dataset_idx], num_workers=num_workers, batch_size=batch_size
        )
        for dataset_idx, dataset in enumerate(datasets)
    }
