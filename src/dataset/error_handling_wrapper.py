from torch.utils.data import IterableDataset
from typing import Iterator, Any
import traceback


class ErrorHandlingDatasetWrapper(IterableDataset):
    """Wrapper that catches exceptions during iteration and skips problematic examples."""
    
    def __init__(self, dataset: IterableDataset):
        super().__init__()
        self.dataset = dataset
    
    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.dataset)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except (EOFError, OSError, KeyError, ValueError, Exception) as e:
                # Log the error and continue to next example
                print(f"Error loading example, skipping: {type(e).__name__}: {e}")
                # Optionally print traceback for debugging
                # traceback.print_exc()
                continue

