import torch
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClipObjEmbedder(torch.nn.Module):
    """Clip Text Embedder to provide label-based embeddings for different object categories"""
    # TODO: 2 clip variants 768 and 512 dimenstional text embedding
    def __init__(self, clip_model:str="ViT-B/32"):
        """Initialize the instance based on clip model variant

        Args:
            clip_model: Variant of clip  
        """
        super(ClipObjEmbedder, self).__init__()
        assert clip_model in ["ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px"], "Use a valid CLIP model variant"
        self.model, self.preprocess = clip.load(clip_model,device=device)
        self.emb_dim = 512

    def forward(self, x):
        """Forward pass
        
        args:
            x: tensor (batch_s,77) Tokenized label (77 is max. token length)
        """

        text_emb = self.model.encode_text(x)
        text_emb /= text_emb.norm(dim=-1, keepdim=True) # normalize
       
        return text_emb