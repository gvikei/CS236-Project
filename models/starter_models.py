import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import *
from torch.nn.utils.rnn import pad_sequence

from utils import *


class BERTEncoder(nn.Module):
    """A pretrained model used to embed text to a 768 dimensional vector.
    """
    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.max_len = 50
        self.embed_size = 768

    def forward(self, text_batch):
        """
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        """
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        device = list(self.parameters())[0].device
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0).to(device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)


class CaptionConditionedGenerativeModel(nn.Module):
    """Base interface that your models should implement.
    """
    def __init__(self, caption_embedding_dim):
        self.caption_embedding_dim = caption_embedding_dim
        super(CaptionConditionedGenerativeModel, self).__init__()

    def forward(self, imgs, captions_embd):
        """
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a GAN.
        """
        loss = None
        other_outputs = {}
        raise NotImplementedError

    def likelihood(self, imgs, captions_embd):
        """
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        """
        raise NotImplementedError

    def sample(self, captions_embd):
        """
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        """
        raise NotImplementedError

