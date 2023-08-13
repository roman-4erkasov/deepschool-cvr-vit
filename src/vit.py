import torch
from torch import nn
# from torchtyping import TensorType, patch_typeguard


class PatchEmbedder(nn.Module):
    def __init__(
        self, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 64
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
        self._embedder = nn.Linear(
            in_features=self.patch_size*self.patch_size*in_channels,
            out_features=self.embed_dim
        )

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
        N, C, H, W = tensor.shape
        n_patches = (H*W)//(self.patch_size*self.patch_size)
        # volume of a pathch
        patch_vol = self.patch_size * self.patch_size * C
        # [N, C, H, W] -> [N, C, H//P, P, W//P, P]
        # for every N,C,W we apply: H -> [H//P, P]
        # for every N,C,H we apply: W -> [W//P, P]
        tensor = tensor.view(
            N,
            C,
            H//self.patch_size,
            self.patch_size,
            W//self.patch_size,
            self.patch_size,
        )
        # [N, C, H//P, P, W//P, P] -> [N, H//P, W//P, P, P]
        tensor = tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        # [N, H//P, W//P, P, P] -> [N, (H//P)*(W//P), P*P]
        tensor = tensor.view(N, n_patches, patch_vol)
        result = self._embedder(tensor)
        return result


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
        self.embed_dim=embed_dim
        self.patch_embedder = PatchEmbedder(
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=self.embed_dim
        )
        n_patches = (image_size*image_size)//(patch_size*patch_size)
        # Вам надо дописать объявление матрицы позиционных эмбеддингов.
        # Помните, что эта матрица - обучаемый параметр!
        self.pos_embeddings = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=0.05,
                size=(1, n_patches + 1, self.embed_dim), # n_patches + CLS token
            ),
        )
        
        self.cls_token = torch.nn.Parameter(
            torch.zeros(1, 1, self.embed_dim),
        )

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
        # raise NotImplementedError
        batch_sz = tensor.size(0)
        # cls_tokens = self.cls_token.repeat((batch_sz, 1, 1))
        # embedded_patches = self.patch_embedder(tensor)
        patch_embeddings = torch.cat(
            (
                self.cls_token.repeat((batch_sz, 1, 1)), 
                self.patch_embedder(tensor)
            ), 
            dim=1
        )
        return patch_embeddings + self.pos_embeddings


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 768, 
        qkv_dim: int = 64, 
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Нужно создать слои для Q, K, V, не забыть про нормализацию
        # на корень из qkv_dim и про дропаут.
        self.wq = torch.nn.Linear(
            in_features=embed_dim,
            out_features=qkv_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            in_features=embed_dim,
            out_features=qkv_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            in_features=embed_dim,
            out_features=qkv_dim,
            bias=False,
        )
        self.scale = qkv_dim ** -0.5
        self.dropout = torch.nn.Dropout(dropout_rate)

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
        q = self.wq(tensor)
        k = self.wk(tensor)
        v = self.wv(tensor)
        # print(f"[ScaledDotProductAttention][forward] {tensor.shape=}")
        # print(f"[ScaledDotProductAttention][forward] {q.shape=} {k.shape=} {v.shape=}")
        result = self.dropout(((q @ k.mT) * self.scale).softmax(dim=-1)) @ v
        # print(f"[ScaledDotProductAttention][forward] {result.shape=}")
        return result
        


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.dropout_rate = dropout_rate
        
        # Надо вспомнить, что в селф-аттеншене участвует несколько голов,
        # и сооздать их соответствующее количество.
        self.attns = nn.ModuleList(
            ScaledDotProductAttention(
                embed_dim=self.embed_dim, 
                qkv_dim=self.qkv_dim, 
                dropout_rate=dropout_rate
            )
            for _ in range(self.n_heads)
        )


        # А тут надо вспомнить, что внутри ViT размерность эмбеддингов
        # не меняется, и поэтому после нескольких голов селф-аттеншена
        # полученные эмбеддинги надо вернуть в их исходную размерность.
        # Конечно же, не забыв про дропаут.
        self.projection = nn.Sequential(
            nn.Linear(self.n_heads*self.qkv_dim, self.embed_dim),
            nn.Dropout(p=self.dropout_rate)
        )
        # print(f"{self.n_heads=}\n{self.qkv_dim=}")

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
        qkv_out = torch.cat(
            [attn(tensor) for attn in self.attns],
            axis=-1
        )
        # print(f"[MultiHeadSelfAttention][forward]{qkv_out.shape=}")
        return self.projection(qkv_out)


class MLP(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        # MLP from paper of ViT (arxiv:2010.11929)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.mlp(tensor)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        mlp_hidden_size: int = 3072,
        attention_dropout_rate: float = 0.1,
        qkv_dim: int = 64,
        n_heads: int = 12,
        mlp_dropout_rate: float = 0.1,
    ):
        super().__init__()
        # print(f"[EncoderBlock]")
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            qkv_dim=qkv_dim,
            n_heads=n_heads,
            dropout_rate=attention_dropout_rate,
        )
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_hidden_size=mlp_hidden_size,
            dropout_rate=mlp_dropout_rate,
        )
    def forward(self, tensor: torch.Tensor):
        msa_out = self.msa(self.norm1(tensor)) + tensor
        return self.mlp(self.norm2(msa_out)) + msa_out
        

class ViT(torch.nn.Module):
    CLS_TOKEN_INDEX = 0
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
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Нужно создать весь энкодер ViT'a, не забыв про LinearProjection.
        self.encoder = nn.Sequential(
            LinearProjection(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            ),
            *[
                EncoderBlock(
                    embed_dim=embed_dim,
                    mlp_hidden_size=mlp_hidden_size,
                    attention_dropout_rate=attention_dropout_rate,
                    qkv_dim=qkv_dim,
                    n_heads=n_heads,
                    mlp_dropout_rate=mlp_dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )

        # и классификационную голову.
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        1) Прогнать через энкодер;
        2) Прогнать через классификационную голову, не забыв, что в
           статье в нее подается только эмбеддинг токена класса.
        """
        # print(f"[ViT][forward] {tensor.shape=}")
        embeddings = self.encoder(tensor)
        # print(f"[ViT][forward] {embeddings.shape=}")
        return self.classifier(embeddings[:,self.CLS_TOKEN_INDEX,:])
