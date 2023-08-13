import torch
from torch import nn


class PatchEmbedder(nn.Module):
    def __init__(
        self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 64
	) -> None:
        """
        Args:
            patch_size: Размер патча в пикселях;
            in_channels: Число каналов у входного изображения;
            embed_dim: Размерность вектора, в который будет преобразован
                один патч.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Добавьте сверточный слой, который преобразует патчи из
        # изображения в векторы.
        self._embedder = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Преобразует батч изображений в батч эмбеддингов
        патчей.

        Args:
            tensor: Батч изображений.
        
        Note:
            На вход приходит некоторый тензор размера (N, C, H, W).
            Нам надо преобразовать его в батч эмбеддингов патчей
            размера (N, H*W, embed_dim)
        """
        raise NotImplementedError


class LinearProjection(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            image_size: Размер исходного изображения;
            patch_size: Размер патча в пикселях;
            in_channels: Число каналов у входного изображения;
            embed_dim: Размерность вектора, в который будет преобразован
                один патч.
        """
        super().__init__()
        self.patch_embedder = PatchEmbedder(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )

        # Вам надо дописать объявление матрицы позиционных эмбеддингов.
        # Помните, что эта матрица - обучаемый параметр!
        self.pos_embeddings = ...
        
        # И не забываем про токен класса.
        self.cls_token = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Надо сделать следующее:
        1) Заэмбеддить патчи изображений в векторы с помощью PatchEmbedder'a;
        2) Добавить к данным эмбеддингам эмбеддинг токена класса;
        3) Сложить с матрицей позиционных эмбеддингов.

        Args:
            tensor: Батч с картинками.
        
        Note:
            На вход идет батч с картинками размера (N, C, H, W).
            На выходе мы должны получить батч эмбеддингов патчей
            и токена класса, сложенных с матрицей позиционных
            эмбеддингов. Размер этого счастья должен быть (N, H*W+1, embed_dim).
        """
        raise NotImplementedError


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, embed_dim: int = 768, qkv_dim: int = 64, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()

        # Нужно создать слои для Q, K, V, не забыть про нормализацию
        # на корень из qkv_dim и про дропаут.
        self.wq = ...
        self.wk = ...
        self.wv = ...
        self.scale = ...
        self.dropout = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Надо получить Q, K, V и аккуратно посчитать аттеншн,
        не забыв про дропаут.

        Args:
            tensor: Батч эмбеддингов патчей.
        
        Note:
            Размер входа: (N, H*W+1, embed_dim).
            Размер выхода: (N, H*W+1, qkv_dim)
        """
        raise NotImplementedError


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Надо вспомнить, что в селф-аттеншене участвует несколько голов,
        # и сооздать их соответствующее количество.
        self.attns = ...

        # А тут надо вспомнить, что внутри ViT размерность эмбеддингов
        # не меняется, и поэтому после нескольких голов селф-аттеншена
        # полученные эмбеддинги надо вернуть в их исходную размерность.
        # Конечно же, не забыв про дропаут.
        self.projection = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        1) Считаем все головы аттеншена;
        2) Проецируем результат в исходную размерность.

        Args:
            tensor: Батч эмбеддингов патчей.
        
        Note:
            Размер входа: (N, H*W+1, embed_dim).
            Размер выхода: (N, H*W+1, embed_dim).
        """
        raise NotImplementedError


class MLP(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Тут должен быть кэжуал многослойный персептрон из статьи.
        self.mlp = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ViT(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1_000,
    ):
        super().__init__()

        # Нужно создать весь энкодер ViT'a, не забыв про LinearProjection.
        self.encoder = ...

        # и классификационную голову.
        self.classifier = ...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        1) Прогнать через энкодер;
        2) Прогнать через классификационную голову, не забыв, что в
           статье в нее подается только эмбеддинг токена класса.
        """
        raise NotImplementedError
