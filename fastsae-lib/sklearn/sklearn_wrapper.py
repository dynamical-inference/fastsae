from datetime import datetime
import os
import pathlib
from typing import Optional

from fastsae.dataset.flexible import create_dataloader
from fastsae.interpret.sae_interpreter import SAEInterpreter
from fastsae.models import SAE as SAEModel
from fastsae.models.backbone import Backbone
from fastsae.models.wrapper import SAEWrapper
from fastsae.training.util import get_scheduler
import numpy as np
import PIL.Image
import shortuuid
import sklearn
import sklearn.utils
import sklearn.utils.validation
import torch
import transformers


class IdentityBackbone(torch.nn.Module):
    """
    Identity backbone for pre-extracted features (numpy input).
    Simply passes through tensor data without any model processing.

    Note: This backbone is primarily used for sklearn compatibility,
    particularly to pass sklearn.utils.estimator_checks.check_estimator()
    tests which expect estimators to work with generic numpy arrays
    rather than specialized data formats like images.
    """

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.eval()  # Always in eval mode since no parameters to train

    def forward_to_z(self, x):
        """Pass through tensor data as-is, with proper sklearn input validation."""
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        elif isinstance(x, (list, tuple)) and len(x) > 0:
            # Handle batch from TensorDataset (unwrap tuple)
            return x[0].to(device=self.device, dtype=self.dtype)
        else:
            x_validated = sklearn.utils.validation.check_array(
                x, accept_sparse=False, ensure_2d=True, allow_nd=True, dtype="numeric"
            )  # This handles _NotAnArray and other sklearn test objects correctly
            return torch.tensor(x_validated, device=self.device, dtype=self.dtype)

    def save(self, save_path):
        """No-op save since no parameters to save."""
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Create an empty marker file to indicate identity backbone
        with open(f"{save_path}/identity_backbone.txt", "w") as f:
            f.write("Identity backbone - no parameters")

    def eval(self):
        """Set to eval mode (no-op since no parameters)."""
        return super().eval()


class IdentityBackboneEncoder:
    """Picklable encoder for IdentityBackbone compatibility."""

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def __call__(self, x):
        # Call backbone.forward_to_z and handle single tensor return
        z = self.wrapper.backbone.forward_to_z(x)
        return self.wrapper.sae.encode(z)


class SAE(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        backbone_layer: Optional[str] = None,
        d_sae_factor: int = 128,
        architecture: str = "vanilla",
        activation: str = "relu",
        device: str = "cuda_if_available",
        sparsity_coefficient: float = 0.00008,
        learning_rate: float = 0.0004,
        scheduler_name: str = "constantwithwarmup",
        warm_up_steps: int = 500,
        use_ghost_grads: bool = True,
        sae_b_dec_init_method: str = "geometric_median",
        stats_top_k_max_samples: int = 100,
        return_pixel_values_only: bool = False,
        total_n_training_samples: int = 1_000_000,
        total_n_checkpoints: int = 10,
        batch_size: int = 256,
        collate: str = "auto",
        random_state: Optional[int] = None,
        _caching: bool = True,
        _verbose: bool = False,
    ):
        """
        Sklearn wrapper for SAEWrapper from wrapper.py.

        Supports four input modes:
        1. Pre-computed feature tensors (torch.Tensor)
        2. HuggingFace Dataset - automatically creates DataLoader internally with HuggingFace backbone models
        3. Raw image data (list) - uses HuggingFace backbone models (e.g., CLIP)
        4. Pre-extracted features (numpy arrays) - uses identity backbone for sklearn compatibility

        # Use tensor directly with SAE
        sae = SAE(backbone_name=None)  # Identity backbone for pre-computed features
        sae.fit(X_train)
        ```

        For loading a saved model:
        ```python
        sae_sklearn = SAE.load(load_dir="path/to/saved_model")
        ```

        Note: The identity backbone mode is specifically designed to pass
        sklearn.utils.estimator_checks.check_estimator() tests.

        Parameters:
        -----------
        backbone_name : str, optional, default=None
            Name of the Hugging Face model to use as backbone.
            If None (default), uses identity backbone for sklearn compatibility with numpy arrays.
        backbone_layer : str or int, optional, default=None
            Layer to hook in the backbone model. Ignored if backbone_name is None.
        d_sae_factor : int, default=128
            Expansion factor for the SAE latent space (d_sae = d_in * d_sae_factor)
        architecture : str, default="vanilla"
            SAE architecture type ("vanilla")
        activation : str, default="relu"
            Activation function ("relu", "gelu", "softplus")
        device : str, default="cuda_if_available"
            Device to run computations on ("cpu", "cuda", or "cuda_if_available")
        sparsity_coefficient : float, default=0.00008
            L1 sparsity coefficient for SAE training
        learning_rate : float, default=0.0004
            Learning rate for SAE training
        scheduler_name : str, default="constantwithwarmup"
            Learning rate scheduler name
        warm_up_steps : int, default=500
            Number of warmup steps for scheduler
        use_ghost_grads : bool, default=True
            Whether to use ghost gradients for dead latent handling
        sae_b_dec_init_method : str, default="geometric_median"
            Method for initializing decoder bias
        stats_top_k_max_samples : int, default=100
            Number of top-k most highly activating samples to compute and store stats for
        return_pixel_values_only : bool, default=False
            Whether to return only pixel values from processor
        total_n_training_samples : int, default=1_000_000
            Total number of training samples
        batch_size : int, default=256
            Batch size for DataLoader creation
        collate : str, default="auto"
            Collate function type ("auto", "default", "images")
        random_state : int, optional, default=None
            Random seed for reproducible results. If None, the estimator is non-deterministic.
        _caching : bool, optional, default=False
            Whether to use cached latents. If True, the latents will be loaded from the cache directory.
        # cache_dir : str, optional, default="./backbone_cache"
        #     Cache directory for the latents.
        _verbose : bool, optional, default=False
            Whether to print verbose output
        """
        params = locals().copy()
        params.pop("self")
        self.__dict__.update(params)
        # self._model_hash = itune._backends.utils.hash_model(self)
        self._run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shortuuid.uuid()[:6]}"
        self._check_wandb()

    @staticmethod
    def _getenv_bool(env_var_name: str, default_value: bool = False) -> bool:
        return os.getenv(env_var_name, str(default_value)).lower() in (
            "true",
            "1",
            "yes",
            "y",
        )

    def _check_wandb(self):
        wandb_enabled = self._getenv_bool("WANDB_ENABLED")

        if wandb_enabled:
            self._wandb_enabled = True
            self._wandb_run_id = self._run_id
        else:
            self._wandb_enabled = False
            self._wandb_run_id = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = self.random_state is None
        return tags

    def fit(self, X, y=None):
        if self._is_dataset(X):
            dataloader, d_in, auto_processor, _ = self._handle_dataset_input(X)
        elif isinstance(X, torch.Tensor):
            dataloader, d_in = self._handle_tensor_input(X)
        elif isinstance(X, list):
            dataloader, d_in = self._handle_list_input(X)
        else:
            dataloader, d_in = self._handle_numpy_input(X)

        self._create_wrapper_architecture(d_in)
        self.n_features_in_ = d_in
        self.set_ckpt_path()
        self._train_sae(dataloader)
        self._set_eval()

        return self

    def _extract_feature_dimensions(self, sample_input, backbone=None):
        if backbone is None:
            backbone = self._create_backbone()

        with torch.no_grad():
            sample_features = backbone.forward_to_z(sample_input)

            # Handle tuple return from IdentityBackbone (for sklearn API test compatibility)
            if isinstance(sample_features, tuple):
                _, sample_features = sample_features

            if len(sample_features.shape) > 2:
                sample_features = sample_features.mean(dim=1)
            d_in = sample_features.shape[-1]

        self.original_dtype_ = np.float32
        return d_in

    def _get_feature_dimensions_from_dataloader(self, dataloader):
        sample_batch = next(iter(dataloader))
        return self._extract_feature_dimensions(sample_batch)

    def _handle_dataset_input(self, X, shuffle=True, return_label=False, return_labels_np=False):
        dataloader_dict = self.get_dataloader(
            X,
            return_processor=True,
            shuffle=shuffle,
            return_label=return_label,
            return_labels_np=return_labels_np,
        )

        dataloader = dataloader_dict["dataloader"]
        auto_processor = dataloader_dict["auto_processor"]
        labels_np = dataloader_dict["labels_np"]

        X_tensor, d_in = (
            dataloader,
            self._get_feature_dimensions_from_dataloader(dataloader),
        )
        return X_tensor, d_in, auto_processor, labels_np

    def _handle_list_input(self, X):
        sample_batch = X[0]
        d_in = self._extract_feature_dimensions(sample_batch)
        return X, d_in

    def _handle_numpy_input(self, X):
        X_tensor, d_in = self._numpy_to_tensor_input(X)
        return X_tensor, d_in

    def _handle_tensor_input(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(X)}")

        if X.dim() < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions, got shape {X.shape}")

        d_in = X.shape[-1]

        return X, d_in

    def _create_tensor(self, data, dtype=torch.float32, device=None):
        if device is None:
            device = self._get_device()
        return torch.tensor(data, dtype=dtype, device=device)

    def _get_sae_dtype(self):
        return self.wrapper_.sae.dtype if hasattr(self.wrapper_.sae, "dtype") else torch.float32

    def _apply_original_dtype(self, result):
        if hasattr(self, "original_dtype_"):
            return result.astype(self.original_dtype_)
        return result

    def _numpy_to_tensor_input(self, X, set_original_dtype=True):
        if set_original_dtype:
            X_validated = sklearn.utils.validation.validate_data(self, X, accept_sparse=False, reset=True)
            self.original_dtype_ = X_validated.dtype
        else:
            X_validated = sklearn.utils.validation.validate_data(self, X, accept_sparse=False, reset=False)

        X_tensor = self._create_tensor(X_validated)
        return X_tensor, X_validated.shape[1]

    def transform(self, X):
        sklearn.utils.validation.check_is_fitted(self, "wrapper_")

        self._set_eval()

        if self._is_dataset(X):
            return self._transform_dataset(X)
        if isinstance(X, torch.Tensor):
            return self._transform_tensor(X)
        if isinstance(X, list):
            return self._transform_iterable(X)
        if isinstance(X, np.ndarray):
            return self._transform_numpy(X)
        return self._transform_default(X)

    def _is_dataset(self, X):
        return hasattr(X, "__iter__") and hasattr(X, "features") and not isinstance(X, (list, np.ndarray))

    def _contains_image_data(self, X):
        if X.dtype == object and len(X) > 0:
            first_element = X.flat[0] if X.size > 0 else None
            if isinstance(first_element, dict) and "image" in first_element:
                return isinstance(first_element["image"], PIL.Image.Image)
        return False

    def _transform_dataset(self, X):
        dataloader = self.get_dataloader(X, shuffle=False)
        return self._transform_iterable(dataloader)

    def _transform_image_array(self, X):
        activations_list = []
        for item in X.flat:
            if isinstance(item, dict) and "image" in item:
                image = item["image"]
                activations_list.append(self._process_activations_batch(image))
            else:
                raise ValueError(f"Expected dictionary with 'image' key, got {type(item)}")

        result = np.vstack(activations_list)
        return self._apply_original_dtype(result)

    def _process_activations_batch(self, batch):
        with torch.no_grad():
            activations = self.wrapper_.encode(batch)
            return activations.cpu().numpy()

    def _transform_iterable(self, X):
        activations_list = []
        for batch in X:
            activations_list.append(self._process_activations_batch(batch))
        result = np.vstack(activations_list)
        return self._apply_original_dtype(result)

    def _transform_tensor(self, X):
        torch_dtype = self._get_sae_dtype()
        if X.dtype != torch_dtype:
            X = X.to(dtype=torch_dtype)

        with torch.no_grad():
            if self.backbone_name is None:
                activations = self.wrapper_.sae.encode(X)
            else:
                activations = self.wrapper_.encode(X)

        result = activations.cpu().numpy()
        return self._apply_original_dtype(result)

    def _transform_numpy(self, X):
        if self._contains_image_data(X):
            return self._transform_image_array(X)
        X_tensor, _ = self._numpy_to_tensor_input(X, set_original_dtype=False)
        return self._transform_tensor(X_tensor)

    def _transform_default(self, X):
        result = self.wrapper_.encode(X).mean(1).detach().cpu().numpy()  # (batch_size, sae_dim), averaged over tokens
        return self._apply_original_dtype(result)

    def inverse_transform(self, X):
        sklearn.utils.validation.check_is_fitted(self, "wrapper_")

        original_shape = None
        if hasattr(X, "shape") and len(X.shape) == 3:
            original_shape = X.shape
            # Reshape to 2D for processing (n_samples*sequence_length, d_sae)
            X = X.reshape(-1, X.shape[-1])

        X = sklearn.utils.check_array(X, accept_sparse=False, dtype="numeric", ensure_2d=True)

        torch_dtype = self._get_sae_dtype()
        X_tensor = self._create_tensor(X, dtype=torch_dtype)

        with torch.no_grad():
            reconstructed = self.wrapper_.sae.decode(X_tensor)

        result = reconstructed.cpu().numpy()

        if original_shape is not None:
            result = result.reshape(original_shape[0], original_shape[1], -1)

        if hasattr(self, "original_dtype_"):
            result = result.astype(self.original_dtype_)
        return result

    def compute_stats(self, X, stats_path=None):
        if self._is_dataset(X):
            dataloader, d_in, auto_processor, labels_np = self._handle_dataset_input(
                X,
                shuffle=False,
                return_labels_np=True,  # NOTE: set shuffle=False and return_labels_np=True when computing stats
            )
        else:
            raise NotImplementedError(f"Expected dataset, got {type(X)}")

        self._set_eval()
        stats_path = stats_path if stats_path is not None else pathlib.Path(self.get_ckpt_path(), "final", "stats")
        self._compute_stats(dataloader, auto_processor, labels_np, save_path=stats_path)

    def _compute_stats(self, dataloader, backbone_processor, labels_np, save_path=None):
        intrp = SAEInterpreter(
            wrapped=self.wrapper_,
            backbone_processor=backbone_processor,
            stats_dataloader=dataloader,
            stats_labels_np=labels_np,
            stats_top_k_max_samples=self.stats_top_k_max_samples,
            _verbose=self._verbose,
        )

        if save_path is not None:
            try:
                save_path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"Stats directory {save_path} already exists. Try loading stats instead.")
                intrp.load_stats(save_path)
                stats = {
                    k: intrp.stats[k].cpu()
                    for k in [
                        "top_idx",
                        "top_label",
                        "top_act",
                        "mean_act",
                        "sparsity",
                        "top_entropy",
                    ]
                }
                self.stats_ = stats
                return

        intrp.compute_stats()
        if save_path is not None:
            intrp.save_stats(save_path)

        stats = {
            k: intrp.stats[k].cpu()
            for k in [
                "top_idx",
                "top_label",
                "top_act",
                "mean_act",
                "sparsity",
                "top_entropy",
            ]
        }
        self.stats_ = stats

    def _train_sae(self, dataloader):
        optimizer = torch.optim.Adam(self.wrapper_.sae.parameters(), lr=self.learning_rate)
        scheduler = get_scheduler(self.scheduler_name, optimizer=optimizer, warm_up_steps=self.warm_up_steps)

        self.wrapper_.train(
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_n_training_samples=self.total_n_training_samples,
            sparsity_coefficient=self.sparsity_coefficient,
            log_to_wandb=self._wandb_enabled,
            wandb_run_id=self._wandb_run_id,
            use_ghost_grads=self.use_ghost_grads,
            sae_b_dec_init_method=self.sae_b_dec_init_method,
            checkpoint_path=self._ckpt_path,
            total_n_checkpoints=self.total_n_checkpoints,
        )

    def set_ckpt_path(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = f"{os.getenv('CHECKPOINT_PATH', './checkpoints')}/{self._run_id}"
        self._ckpt_path = ckpt_path

    def get_ckpt_path(self):
        return self._ckpt_path

    def get_dataloader(
        self,
        X,
        return_processor=False,
        shuffle=True,
        return_label: Optional[bool] = False,
        return_labels_np: Optional[bool] = False,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
    ):
        auto_processor = None
        processor = None

        if self.backbone_name:
            auto_processor = transformers.AutoProcessor.from_pretrained(self.backbone_name)
            processor = lambda img: auto_processor(images=img, text="a photo", return_tensors="pt")  # noqa: E731

        if num_workers is None:
            num_workers = int(os.getenv("SAE_DATALOADER_NUM_WORKERS", 0))
        if pin_memory is None:
            pin_memory = self._getenv_bool("SAE_DATALOADER_PIN_MEMORY", True)
        if persistent_workers is None:
            persistent_workers = self._getenv_bool("SAE_DATALOADER_PERSISTENT_WORKERS", True)

        dataloader_tuple = create_dataloader(
            dataset=X,
            processor=processor,
            return_pixel_values_only=self.return_pixel_values_only,
            return_label=return_label,
            return_labels_np=return_labels_np,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate=self.collate,
        )

        if isinstance(dataloader_tuple, tuple) and len(dataloader_tuple) == 2:
            dataloader = dataloader_tuple[0]
            labels_np = dataloader_tuple[1]
        else:
            dataloader = dataloader_tuple
            labels_np = None
        return {
            "dataloader": dataloader,
            "auto_processor": auto_processor,
            "labels_np": labels_np,
        }

    def _create_backbone(self):
        # Use identity backbone if no backbone name is specified or if numpy input is detected
        if self.backbone_name is None:
            torch_dtype = torch.float32
            return IdentityBackbone(device=self._get_device(), dtype=torch_dtype)

        return Backbone(
            model_hf_name=self.backbone_name,
            hook_target_name=self.backbone_layer,
            device=self._get_device(),
        )

    def _create_sae(self, d_in):
        return SAEModel(
            d_in=d_in,
            d_sae_factor=self.d_sae_factor,
            architecture=self.architecture,
            activation_name=self.activation,
            device=self._get_device(),
            seed=self.random_state,
        )

    def _create_wrapper_architecture(self, d_in):
        backbone = self._create_backbone()
        sae_instance = self._create_sae(d_in)
        self.wrapper_ = SAEWrapper(backbone=backbone, sae=sae_instance, _caching=self._caching)

        if isinstance(backbone, IdentityBackbone):
            self.wrapper_.encode = IdentityBackboneEncoder(self.wrapper_)

    @classmethod
    def load(cls, load_dir: pathlib.Path, sae_load_name: Optional[str] = None, **kwargs):
        backbone_load_dir = pathlib.Path(load_dir, "backbone") if load_dir is not None else None
        backbone = Backbone.load(backbone_load_dir)

        sae_load_name = "final" if sae_load_name is None else sae_load_name
        sae_load_dir = pathlib.Path(load_dir, sae_load_name) if load_dir is not None else None
        sae_instance = SAEModel.load(sae_load_dir)

        config = {
            "backbone_name": backbone.model_hf_name,
            "backbone_layer": backbone.hook_target_name,
            "d_sae_factor": sae_instance.d_sae_factor,
            "architecture": sae_instance.architecture,
            "activation": sae_instance.activation,
            "device": sae_instance.device,
            **kwargs,
        }

        cls_instance = cls(**config)
        cls_instance.wrapper_ = SAEWrapper(backbone=backbone, sae=sae_instance)
        cls_instance.set_ckpt_path(load_dir)
        return cls_instance

    def load_state_dict(self, state_dict):
        dummy_features = state_dict.get("n_features_in_", 10)
        self._create_wrapper_architecture(dummy_features)
        self.wrapper_.load_state_dict(state_dict["wrapper_state_dict"])

        if "original_dtype_" in state_dict and state_dict["original_dtype_"] is not None:
            self.original_dtype_ = state_dict["original_dtype_"]

        if "n_features_in_" in state_dict and state_dict["n_features_in_"] is not None:
            self.n_features_in_ = state_dict["n_features_in_"]

        if "stats_" in state_dict and state_dict["stats_"] is not None:
            self.stats_ = state_dict["stats_"]

    def state_dict(self):
        return {
            "wrapper_state_dict": self.wrapper_.state_dict(),
            "n_features_in_": getattr(self, "n_features_in_", None),
            "original_dtype_": getattr(self, "original_dtype_", None),
            "stats_": getattr(self, "stats_", None),
        }

    def _get_device(self):
        if self.device == "cuda_if_available":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _set_eval(self):
        self.wrapper_.backbone.eval()
        self.wrapper_.sae.eval()
        self.wrapper_.sae.eval()
